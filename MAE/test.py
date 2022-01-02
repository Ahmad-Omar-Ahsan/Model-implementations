from models import ViT
from models import MAE
import torch

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)
mae = MAE(
    encoder = v,
    masking_ratio = 0.75,   
    decoder_dim = 512,      
    decoder_depth = 6       
)
images = torch.randn(8, 3, 256, 256)

loss = mae(images)
loss.backward()
