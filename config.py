# Hyperparameters
weight_decay = 0.0001
batch_size = 128
num_epochs = 10 
dropout_rate = 0.2
image_size = 256
patch_size = 8
num_patches = (image_size // patch_size) ** 2
embedding_dim = 256
num_blocks = 4
num_classes = 10
input_shape = (256, 256, 3)

# Data
data_dir = 'photozilla'

# Learning rates
mlpmixer_lr = 0.005
fnet_lr = 0.001
gmlp_lr = 0.003