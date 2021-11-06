import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm2d

class DCGAN(nn.Module):

    def __init__(self, latent_dim : int = 100, num_feat_maps_gen : int = 64, num_feat_maps_dis : int = 64, color_channels : int = 3) -> nn.Module:
        """[summary]

        Args:
            latent_dim (int, optional): Size of latent dimension. Defaults to 100.
            num_feat_maps_gen (int, optional): Size of feature map for generator. Defaults to 64.
            num_feat_maps_gen (int, optional): Size of feature map for discriminator. Defaults to 64.
            color_channels (int, optional): Number of channels. Defaults to 3.

        Returns:
            nn.Module: DCGAN network
        """
        super.__init__()

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, num_feat_maps_gen*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_feat_maps_gen*8),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(num_feat_maps_gen*8, num_feat_maps_gen*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat_maps_gen*4),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(num_feat_maps_gen*4, num_feat_maps_gen*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat_maps_gen*2),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(num_feat_maps_gen*2, num_feat_maps_gen, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat_maps_gen),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(num_feat_maps_gen, color_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()

        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(color_channels, num_feat_maps_dis, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(num_feat_maps_dis, num_feat_maps_dis * 2, kernel_size = 4, stride = 2, padding=1, baise=False),
            nn.BatchNorm2d(num_feat_maps_dis*2),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(num_feat_maps_dis*2, num_feat_maps_dis*4,kernel_size=4, stride=2, padding=1, bias=False),        
            nn.BatchNorm2d(num_feat_maps_dis*4),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(num_feat_maps_dis*4, num_feat_maps_dis*8, kernel_size=4, stride=2, padding=1, bias=False),        
            nn.BatchNorm2d(num_feat_maps_dis*8),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(num_feat_maps_dis*8, 1, kernel_size=4, stride=1, padding=0),
            nn.Flatten()
        )

    
    def generator_forward(self, z):
        img = self.generator(z)
        return img

    def discriminator_forward(self, img):
        logits = self.discriminator(img)
        return logits


        