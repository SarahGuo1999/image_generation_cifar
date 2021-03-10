import torch

def quantize(A):
	'''
	quantize matrix A
	assert there are positive and negative values in matrix A
	'''
	# find the top and bottom clip_ratio number
	top_cut = A.max().item()
	bottom_cut = A.min().item()
	scale = max(top_cut, -bottom_cut)

	# map scale to INT8_MAX and -scale to -INT8_MAX
	QA = (A * 127 / scale).round()
	return QA, scale

def quantized_matmul(A, B):
	QA, _ = quantize(A)
	QB, _ = quantize(B)
	return torch.matmul(QA, QB)

def calc_topk_matrix_after_masking(A, B, masked_qprod, k):
	_, idx = torch.topk(masked_qprod, k)

	prod = torch.matmul(A, B)
	val = prod.gather(index=idx, dim=-1) # mask prod by idx

	out = torch.full_like(prod, float('-inf'))
	out.scatter_(src=val, index=idx, dim=-1) # fills the values according to idx
	return out