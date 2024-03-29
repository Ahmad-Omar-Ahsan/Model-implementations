import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, down : bool = True, use_act : bool = True, **kwargs):
        """ Initializes convolutional block for generator

        Args:
            in_channels (int):  Number of input channels. 
            out_channels (int): Number of output channels
            down (bool, optional): Boolean for downsample. Defaults to True.
            use_act (bool, optional): Boolean for using activation. Defaults to True.
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) if down else 
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        ) 
    
    def forward(self, x : torch.Tensor):
        """ Forward function for ConvBlock

        Args:
            x (torch.Tensor): Input tensor
        """
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels : int):
        """ Initializes Residual Block

        Args:
            channels (int): Number of channels
        """
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x : torch.Tensor):
        """ Forward function for ResidualBlock

        Args:
            x (torch.Tensor): Input tensor
        """
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels : int, num_features : int = 64, num_residuals : int = 9):
        """ Initializes Generator 

        Args:
            img_channels (int): Number of input channels
            num_features (int, optional): Number of features. Defaults to 64.
            num_residuals (int, optional): Number of ResBlocks. Defaults to 9.
        """
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.Sequential(
            ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
            ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.Sequential(
            ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
    
    def forward(self, x : torch.Tensor):
        """ Forward function for generator

        Args:
            x (torch.Tensor): Input tensor
        """
        x = self.initial(x)
        x = self.down_blocks(x)
        x = self.res_blocks(x)
        x = self.up_blocks(x)
        return torch.tanh(self.last(x))



