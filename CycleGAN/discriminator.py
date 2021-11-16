import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, stride:int) -> nn.Module:
        """ Initializes block for discriminator

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): number of strides

        Returns:
            nn.Module: returns Block module
        """
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x : torch.Tensor):
        """ Forward function for Block

        Args:
            x (torch.Tensor): [description]
        """
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels : int = 3, features : list = [64, 128, 256, 512]) -> nn.Module:
        """[summary]

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
            features (list, optional): List of channels in the discriminator block. Defaults to [64, 128, 256, 512].

        Returns:
            nn.Module: Returns discriminator block
        """
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
        self.model = nn.Sequential(*layers)

    def forward(self, x : torch.Tensor):
        """ Forward function for discriminator

        Args:
            x (torch.Tensor): Input tensor
        """
        x = self.initial(x)
        return torch.sigmoid(self.model(x))




