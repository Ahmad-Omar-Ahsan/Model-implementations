import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim : int, hidden_dim : int, dropout : float = 0.0) -> nn.Module:
        """Feed forward block

        Args:
            dim (int): Input dimension
            hidden_dim (int): Dimension of dense layers
            dropout (float, optional): Dropout probability. Defaults to 0.0.

        Returns:
            nn.Module: Feed forward layer
        """
        super(FeedForward,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        """ Forward function for feed forward layer

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim : int, heads : int = 8, dim_head : int = 64, dropout : float = 0.0) -> nn.Module:
        """Attention layer

        Args:
            dim (int): Input dimension 
            heads (int, optional): Number of heads for multiheaded self attention. Defaults to 8.
            dim_head (int, optional): Dimension of query, key, value matrices. Defaults to 64.
            dropout (float, optional): Dropout probability . Defaults to 0.0.

        Returns:
            nn.Module: Attention module
        """
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out =  not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """ Forward function for attention layer

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim : int, depth : int, heads : int, dim_head : int, mlp_dim : int, dropout = 0.0) -> nn.Module:
        """ Encoder block for ViT

        Args:
            dim (int): Input dimension.
            depth (int): Number of encoder blocks.
            heads (int): Number of heads.
            dim_head (int): Size of q,k,v vectors.
            mlp_dim (int): Size of dimension in feed forward network.
            dropout (float, optional): Dropout prob. Defaults to 0.0.

        Returns:
            nn.Module: Transformer module
        """
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim,mlp_dim,dropout)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                self.attn,
                self.ff
            ]))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """ Forward function for transformers

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        for attention, feed_forward in self.layers:
            x = attention(self.norm(x))  + x
            x = feed_forward(self.norm(x)) + x
        return x        


class ViT(nn.Module):
    def __init__(
        self,
        image_size : int,
        patch_size : int,
        num_classes : int,
        dim : int,
        depth : int,
        heads : int,
        mlp_dim : int,
        pool : str = 'cls',
        channels : int = 3,
        dim_head : int = 64,
        dropout : float = 0.0, 
        emb_dropout : float = 0.0
    ) -> nn.Module:
        """[summary]

        Args:
            image_size (int): Size of images.
            patch_size (int): Size of patches.
            num_classes (int): Number of classes.
            dim (int): Input dimension.
            depth (int): Number of transformer blocks.
            heads (int): Number of heads.
            mlp_dim (int): Dimension for feed forward network.
            pool (str, optional): Pool type. Defaults to 'cls'.
            channels (int, optional): Number of channels. Defaults to 3.
            dim_head (int, optional): Size of Q,K,V. Defaults to 64.
            dropout (float, optional): Dropout prob. Defaults to 0.0.
            emb_dropout (float, optional): Embedding dropout prob. Defaults to 0.0.

        Returns:
            nn.Module: ViT module.
        """
        super(ViT, self).__init__()
        image_height, image_width = (image_size,image_size) if isinstance(image_size, int) else image_size
        patch_height, patch_width = (patch_size,patch_size) if isinstance(patch_size, int) else patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, "Image dimensions must be divisible by patch dimensions"

        num_patches = (image_width * image_height) // (patch_height * patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim)
        )


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ) 

    def forward(self, img:torch.Tensor) -> torch.Tensor:
        """ Forward function for Vi.t

        Args:
            img (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor
        """
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        return self.mlp_head(x)