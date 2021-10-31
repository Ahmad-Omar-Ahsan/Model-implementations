import torch
import torch.nn as nn
from math import ceil

class CNNBlock(nn.Module):
    """ CNN block in the EfficientNet architecture
        CNN -> bn -> silu

    Args:
        nn (Module): Base class for all neural network modules.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        """Initializes CNN block

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): size of kernel
            stride (int): stride value
            padding (int): padding value
            groups (int, optional): controls the connections between inputs and outputs. 
                                    in_channels and out_channels must both be divisible by groups. 
                                    Defaults to 1. For depthwise convolution set groups to in_channels
        """
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        """Computes the forward propagation for a tensor x

        Args:
            x (torch.float32): input tensor

        Returns:
            [torch.float32]: output tensor
        """
        return self.silu(self.bn(self.cnn(x)))
    



class SqueezeExcitation(nn.Module):
    """Implementation of squeeze and excitation block to capture interchannel dependencies explicitly. 

    Args:
        nn (Module): Base class for all neural network modules.
    """
    def __init__(self, in_channels, reduced_dim):
        """Initializes SqueezeExcitation block

        Args:
            in_channels (int): number of input channels 
            reduced_dim (int): size of reduced dimension, this is provided to reduce model complexity 
        """
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels, reduced_dim, 1),
        nn.SiLU(),
        nn.Conv2d(reduced_dim, in_channels, 1),
        nn.Sigmoid()
        )
    def forward(self, x):
        """Computes the forward propagation for a tensor x

        Args:
            x (torch.float32): input tensor

        Returns:
            [torch.float32]: output tensor
        """
        return x * self.se(x)




class InvertedResidualBlock(nn.Module):
    """ This block implements the inverted residual block similar to mobilenetv2

    Args:
        nn (Module): Base class for all neural network modules.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction=4, survival_prob=0.8):
        """ Initializes inverted residual block

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): size of kernel
            stride (int): stride size
            padding (int): padding size
            expand_ratio (int): expansion ratio
            reduction (int, optional): number to reduce the channels by for squeeze excitation block. Defaults to 4.
            survival_prob (float, optional): survival probability for stochastic depth. Defaults to 0.8.
        """
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)
        
        if self.expand:
            self.expand_conv = CNNBlock(
            in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
            
        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
    def stochastic_depth(self, x):
        """Implements stochastic depth which is used to shrink the depth of the network during training

        Args:
            x (torch.float32): input tensor

        Returns:
            [torch.float32]: output tensor
        """
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        """Computes the forward propagation for a tensor x

        Args:
            x (torch.float32): input tensor

        Returns:
            [torch.float32]: output tensor
        """
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)




class EfficientNet(nn.Module):
    """ Class for implementing EfficientNet architecture. 

    Args:
        nn (Module): Base class for all neural network modules.
    """
    def __init__(self, version, num_classes, base_model,phi_values):
        """ Initializes the efficientnet model

        Args:
            version (string): version of the model e.g. b0 - b6
            num_classes (int): number of classes
            base_model (list): list of values for the inverted residual blocks
            phi_values (dict): tuple of phi value, resolution, drop rate
        """
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )
        self.base_model = base_model
        self.phi_values = phi_values

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        """ Calculates depth factor, drop rate and width factor for the model

        Args:
            version (string): version of the model e.g. b0 - b6
            alpha (float, optional): Constant for depth factor. Defaults to 1.2.
            beta (float, optional): Constant for width factor. Defaults to 1.1.

        Returns:
            [tuple]: the calculated factor values 
        """
        phi, res, drop_rate = self.phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        """Initialize the inverted residual blocks with respect to depth factors, width factors and resolution 

        Args:
            width_factor (float): width factor value
            depth_factor (float): depth factor value
            last_channels (int): final channel value

        Returns:
            [nn.Sequential]: List of layers
        """
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in self.base_model:
            out_channels = 4*ceil(int(channels*width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride = stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size//2, # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                    )
                )
                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, x):
        """Computes the forward propagation for a tensor x

        Args:
            x (torch.float32): input tensor

        Returns:
            [torch.float32]: output tensor
        """
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))
