# image_generation_cifar

git clone https://github.com/SarahGuo1999/image_generation_cifar.git
cd image_generation_cifar/image_generation_cifar/
python3 train* --doc gzy --config transformer_dmol.yml [--sample]
transformer_dmol.yml和transformer_dmol——cifar_lsh.yml的区别只在batch size
attention可以改image_transformer_lsh.py的第270-276行。
