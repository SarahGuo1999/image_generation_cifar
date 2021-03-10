import sys
import os
import math
import torch
import torch.nn as nn
from torch.nn import Identity
import torch.nn.functional as F
from torch.autograd import Function
from functools import partial, reduce, wraps
from itertools import chain
from operator import mul

from local_attention import LocalAttention
from axial_positional_embedding import AxialPositionalEmbedding
from product_key_memory import PKM
from reversible import ReversibleSequence

file_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(file_dir)
print(file_dir)
print(parent_dir)
sys.path.append(parent_dir)
from two_steps_quantized_matmul import quantized_matmul, calc_topk_matrix_after_masking

# sys.path.append(os.path.join(file_dir, 'reformer_pytorch'))
sys.path.append('/home/gzy/anaconda3/envs/pytorch/lib/python3.8/site-packages/reformer_pytorch/')
from reformer_pytorch import *

def default(val, default_val):
    return default_val if val is None else val

class QuantizedQKLSHAttention(LSHAttention):
	def __init__( self,
				  dropout = 0.,
				  bucket_size = 64,
				  n_hashes = 8,
				  causal = False,
				  allow_duplicate_attention = True,
				  attend_across_buckets = True,
				  rehash_each_round = True,
				  drop_for_hash_rate = 0.0,
				  random_rotations_per_head = False,
				  return_attn = False,
				  K = None):
		super().__init__()
		self.K = K

	def forward(self, qk, v, query_len = None, input_mask = None, input_attn_mask = None, **kwargs):
		batch_size, seqlen, dim, device = *qk.shape, qk.device

		query_len = default(query_len, seqlen)
		is_reverse = kwargs.pop('_reverse', False)
		depth = kwargs.pop('_depth', None)

		assert seqlen % (self.bucket_size * 2) == 0, f'Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}'

		n_buckets = seqlen // self.bucket_size
		buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training)

		# We use the same vector as both a query and a key.
		assert int(buckets.shape[1]) == self.n_hashes * seqlen

		total_hashes = self.n_hashes

		ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
		buckets_and_t = seqlen * buckets + (ticker % seqlen)
		buckets_and_t = buckets_and_t.detach()

		# Hash-based sort ("s" at the start of variable names means "sorted")
		sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
		_, undo_sort = sticker.sort(dim=-1)
		del ticker

		sbuckets_and_t = sbuckets_and_t.detach()
		sticker = sticker.detach()
		undo_sort = undo_sort.detach()

		st = (sticker % seqlen)
		sqk = batched_index_select(qk, st)
		sv = batched_index_select(v, st)

		# Split off a "bin" axis so that attention only occurs within chunks.
		chunk_size = total_hashes * n_buckets
		bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
		bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
		bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))

		# Hashing operates on unit-length vectors. Unnormalized query vectors are
		# fine because they effectively provide a learnable temperature for the
		# attention softmax, but normalizing keys is needed so that similarity for
		# the purposes of attention correctly corresponds to hash locality.
		bq = bqk
		bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

		# Allow each chunk to attend within itself, and also one chunk back. Chunk
		# boundaries might occur in the middle of a sequence of items from the
		# same bucket, so this increases the chances of attending to relevant items.
		def look_one_back(x):
			x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
			return torch.cat([x, x_extra], dim=2)

		bk = look_one_back(bk)
		bv = look_one_back(bv)
		bkv_t = look_one_back(bkv_t)

		# Dot-product attention.
		dots = quantized_matmul(bq, bk.transpose(2, 3)) * (dim ** -0.5)
		# dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)
		masked_value = max_neg_value(dots)

		# Mask for post qk attention logits of the input sequence
		if input_attn_mask is not None:
			input_attn_mask = F.pad(input_attn_mask, (0, seqlen - input_attn_mask.shape[-1], 0, seqlen - input_attn_mask.shape[-2]), value=True)
			dot_attn_indices = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
			input_attn_mask = input_attn_mask.reshape(batch_size, -1)
			dot_attn_indices = dot_attn_indices.reshape(batch_size, -1)
			mask = input_attn_mask.gather(1, dot_attn_indices).reshape_as(dots)
			dots.masked_fill_(~mask, masked_value)
			del mask

		# Input mask for padding in variable lengthed sequences
		if input_mask is not None:
			input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), value=True)
			mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
			mkv = look_one_back(mq)
			mask = mq[:, :, :, None] * mkv[:, :, None, :]
			dots.masked_fill_(~mask, masked_value)
			del mask

		# Causal masking
		if self.causal:
			mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
			if seqlen > query_len:
				mask = mask & (bkv_t[:, :, None, :] < query_len)
			dots.masked_fill_(mask, masked_value)
			del mask

		# Mask out attention to self except when no other targets are available.
		self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
		dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
		del self_mask

		# Mask out attention to other hash buckets.
		if not self._attend_across_buckets:
			bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t // seqlen, (batch_size, chunk_size, -1))
			bkv_buckets = look_one_back(bkv_buckets)
			bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
			dots.masked_fill_(bucket_mask, masked_value)
			del bucket_mask

		# Don't double-count query-key pairs across multiple rounds of hashing.
		# There are two possible strategies here. (1) The default is to count how
		# many times a query-key pair is repeated, and to lower its log-prob
		# correspondingly at each repetition. (2) When hard_k is set, the code
		# instead masks all but the first occurence of each query-key pair.
		if not self._allow_duplicate_attention:
			locs1 = undo_sort // bq_t.shape[-1]
			locs2 = (locs1 + 1) % chunk_size
			if not self._attend_across_buckets:
				locs1 = buckets * chunk_size + locs1
				locs2 = buckets * chunk_size + locs2
			locs = torch.cat([
				torch.reshape(locs1, (batch_size, total_hashes, seqlen)),
				torch.reshape(locs2, (batch_size, total_hashes, seqlen)),
			], 1).permute((0, 2, 1))

			slocs = batched_index_select(locs, st)
			b_locs = torch.reshape(slocs, (batch_size, chunk_size, -1, 2 * total_hashes))

			b_locs1 = b_locs[:, :, :, None, :total_hashes]

			bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, total_hashes))
			bq_locs = torch.reshape(bq_locs, b_locs.shape)
			bkv_locs = look_one_back(b_locs)

			dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
			# for memory considerations, chunk summation of last dimension for counting duplicates
			dup_counts = chunked_sum(dup_counts, chunks=(total_hashes * batch_size))
			dup_counts = dup_counts.detach()
			assert dup_counts.shape == dots.shape
			dots = dots - torch.log(dup_counts + 1e-9)
			del dup_counts

		dots = calc_topk_matrix_after_masking(bq, bk.transpose(2, 3), dots, self.K)

		# Softmax.
		dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
		dots = torch.exp(dots - dots_logsumexp).type_as(dots)
		dropped_dots = self.dropout(dots)

		bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
		so = torch.reshape(bo, (batch_size, -1, dim))
		slogits = torch.reshape(dots_logsumexp, (batch_size, -1,))

		# unsort logits
		o = batched_index_select(so, undo_sort)
		logits = slogits.gather(1, undo_sort)

		o = torch.reshape(o, (batch_size, total_hashes, seqlen, dim))
		logits = torch.reshape(logits, (batch_size, total_hashes, seqlen, 1))

		if query_len != seqlen:
			query_slice = (slice(None), slice(None), slice(0, query_len))
			o, logits = o[query_slice], logits[query_slice]

		probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
		out = torch.sum(o * probs, dim=1)

		attn = torch.empty(0, device=device)

		# return unsorted attention weights
		if self._return_attn:
			attn_unsort = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
			attn_unsort = attn_unsort.view(batch_size * total_hashes, -1).long()
			unsorted_dots = torch.zeros(batch_size * total_hashes, seqlen * seqlen, device=device)
			unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
			del attn_unsort
			unsorted_dots = unsorted_dots.reshape(batch_size, total_hashes, seqlen, seqlen)
			attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)

		# return output, attention matrix, and bucket distribution
		return out, attn, buckets

class QuantizedQKLSHSelfAttention(LSHSelfAttention):
	def __init__(self, dim, heads = 8, bucket_size = 64, n_hashes = 8, causal = False, dim_head = None, attn_chunks = 1, random_rotations_per_head = False, attend_across_buckets = True, allow_duplicate_attention = True, num_mem_kv = 0, one_value_head = False, use_full_attn = False, full_attn_thres = None, return_attn = False, post_attn_dropout = 0., dropout = 0., n_local_attn_heads = 0, K = None, **kwargs):
		super(LSHSelfAttention, self).__init__()  # only call `nn.Module` init
		assert dim_head or (dim % heads) == 0, 'dimensions must be divisible by number of heads'
		assert n_local_attn_heads < heads, 'local attention heads must be less than number of heads'

		dim_head = default(dim_head, dim // heads)
		dim_heads = dim_head * heads

		self.dim = dim
		self.heads = heads
		self.dim_head = dim_head
		self.attn_chunks = default(attn_chunks, 1)

		self.v_head_repeats = (heads if one_value_head else 1)
		v_dim = dim_heads // self.v_head_repeats

		self.toqk = nn.Linear(dim, dim_heads, bias = False)
		self.tov = nn.Linear(dim, v_dim, bias = False)
		self.to_out = nn.Linear(dim_heads, dim)

		self.bucket_size = bucket_size
		self.lsh_attn = QuantizedQKLSHAttention(bucket_size=bucket_size, n_hashes=n_hashes, causal=causal, random_rotations_per_head=random_rotations_per_head, attend_across_buckets = attend_across_buckets,  allow_duplicate_attention = allow_duplicate_attention, return_attn = return_attn, dropout = dropout, K = K, **kwargs)
		self.full_attn = FullQKAttention(causal=causal, dropout=dropout)
		self.post_attn_dropout = nn.Dropout(post_attn_dropout)

		self.use_full_attn = use_full_attn
		self.full_attn_thres = default(full_attn_thres, bucket_size)

		self.num_mem_kv = num_mem_kv
		self.mem_kv = nn.Parameter(torch.randn(1, num_mem_kv, dim, requires_grad=True)) if num_mem_kv > 0 else None

		self.n_local_attn_heads = n_local_attn_heads
		self.local_attn = LocalAttention(window_size=bucket_size * 2, causal=causal, dropout=dropout, shared_qk=True, look_forward=(1 if not causal else 0))

		self.callback = None

class QuantizedQKReformer(Reformer):
	def __init__(self, dim, depth, max_seq_len, heads = 8, dim_head = None, bucket_size = 64, n_hashes = 8, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., ff_dropout = 0., ff_activation = None, ff_mult = 4, ff_glu = False, post_attn_dropout = 0., layer_dropout = 0., lsh_attend_across_buckets = True, lsh_allow_duplicate_attention = True, random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, use_rezero = False, use_full_attn = False, full_attn_thres = 0, reverse_thres = 0, num_mem_kv = 0, one_value_head = False, n_local_attn_heads = 0, pkm_layers = tuple(), pkm_num_keys = 128, Ks = None):
		super(Reformer, self).__init__()  # only call `nn.Module` init
		self.dim = dim
		self.depth = depth

		self.bucket_size = bucket_size
		self.num_mem_kv = num_mem_kv

		self.twin_attention = twin_attention
		self.full_attn_thres = full_attn_thres

		get_attn = lambda K: QuantizedQKLSHSelfAttention(dim, heads, bucket_size, n_hashes, causal = causal, dim_head = dim_head, dropout = lsh_dropout, post_attn_dropout = post_attn_dropout, attn_chunks = attn_chunks, allow_duplicate_attention = lsh_allow_duplicate_attention, attend_across_buckets = lsh_attend_across_buckets, random_rotations_per_head = random_rotations_per_head, num_mem_kv = num_mem_kv, use_full_attn = use_full_attn, full_attn_thres = full_attn_thres, one_value_head = one_value_head, n_local_attn_heads = n_local_attn_heads, K = K)
		get_ff = lambda: Chunk(ff_chunks, FeedForward(dim, dropout = ff_dropout, activation = ff_activation, mult = ff_mult, glu = ff_glu), along_dim = -2)
		get_pkm = lambda: PKM(dim, num_keys = pkm_num_keys)

		if weight_tie:
			get_attn, get_ff, get_pkm = map(cache_fn, (get_attn, get_ff, get_pkm))

		blocks = []

		norm_type = ScaleNorm if use_scale_norm else nn.LayerNorm

		residual_fn_wrapper = ReZero if use_rezero else partial(PreNorm, norm_type, dim)

		for ind in range(depth):
			layer_num = ind + 1
			use_pkm = layer_num in cast_tuple(pkm_layers)
			parallel_net = None

			attn = get_attn(Ks[ind])

			if use_pkm:
				parallel_net = get_pkm()
			elif twin_attention:
				parallel_net = get_attn(Ks[ind])
			else:
				parallel_net = get_ff()

			f = residual_fn_wrapper(attn)
			g = residual_fn_wrapper(parallel_net)

			blocks.append(nn.ModuleList([f, g]))

		self.layers = ReversibleSequence(nn.ModuleList(blocks), layer_dropout = layer_dropout, reverse_thres = reverse_thres, send_signal = True)

class QuantizedQKReformerLM(ReformerLM):
	def __init__(self, num_tokens, dim, depth, max_seq_len, heads = 8, dim_head = None, bucket_size = 64, n_hashes = 4, ff_chunks = 100, attn_chunks = 1, causal = False, weight_tie = False, lsh_dropout = 0., ff_dropout = 0., ff_mult = 4, ff_activation = None, ff_glu = False, post_attn_dropout = 0., layer_dropout = 0., random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, use_rezero = False, use_full_attn = False, full_attn_thres = 0, reverse_thres = 0, num_mem_kv = 0, one_value_head = False, emb_dim = None, return_embeddings = False, weight_tie_embedding = False, fixed_position_emb = False, absolute_position_emb = False, axial_position_shape = None, n_local_attn_heads = 0, pkm_layers = tuple(), pkm_num_keys = 128, Ks = None):
		assert len(Ks) == depth
		super(ReformerLM, self).__init__() # only call `nn.Module` init
		emb_dim = default(emb_dim, dim)
		self.max_seq_len = max_seq_len

		self.token_emb = nn.Embedding(num_tokens, emb_dim)

		self.to_model_dim = Identity() if emb_dim == dim else nn.Linear(emb_dim, dim)

		if absolute_position_emb:
			self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len)
		elif fixed_position_emb:
			self.pos_emb = FixedPositionalEmbedding(emb_dim)
		else:
			axial_position_shape = default(axial_position_shape, (math.ceil(max_seq_len / bucket_size), bucket_size))
			self.pos_emb = AxialPositionalEmbedding(emb_dim, axial_position_shape)

		self.reformer = QuantizedQKReformer(dim, depth, max_seq_len, heads = heads, dim_head = dim_head, bucket_size = bucket_size, n_hashes = n_hashes, ff_chunks = ff_chunks, attn_chunks = attn_chunks, causal = causal, weight_tie = weight_tie, lsh_dropout = lsh_dropout, ff_mult = ff_mult, ff_activation = ff_activation, ff_glu = ff_glu, ff_dropout = ff_dropout, post_attn_dropout = 0., layer_dropout = layer_dropout, random_rotations_per_head = random_rotations_per_head, twin_attention = twin_attention, use_scale_norm = use_scale_norm, use_rezero = use_rezero, use_full_attn = use_full_attn, full_attn_thres = full_attn_thres, reverse_thres = reverse_thres, num_mem_kv = num_mem_kv, one_value_head = one_value_head, n_local_attn_heads = n_local_attn_heads, pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys, Ks = Ks)
		self.norm = nn.LayerNorm(dim)

		if return_embeddings:
			self.out = Identity()
			return

		self.out = nn.Sequential(
			nn.Linear(dim, emb_dim) if emb_dim != dim else Identity(),
			nn.Linear(emb_dim, num_tokens) if not weight_tie_embedding else MatrixMultiply(self.token_emb.weight, transpose=True, normalize=True)
		)