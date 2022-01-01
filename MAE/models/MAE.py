import torch
from torch import nn 
import torch.nn.functional as F
from einops import repeat

from ViT import Transformer


class MAE(nn.Module):
    def __init__(
        self,
        encoder : nn.Module,
        decoder_dim : int,
        masking_ratio : float = 0.75,
        decoder_depth : int = 1,
        decoder_heads : int = 8,
        decoder_dim_head : int = 64
    ) -> torch.Tensor:
        """ The mask autoencoder has an ViT encoder block as an encoder and decoder. The encoder block
        receives unmasked patches to encode. The deccoder receives both unamsked and masked tokens and outputs the pixel values for each patch.
        This block returns the mse loss.

        Args:
            encoder (nn.Module): Encoder block.
            decoder_dim (int): Decoder dimension.
            masking_ratio (float, optional): Masking ratio. Defaults to 0.75.
            decoder_depth (int, optional): Number of decoder blocks. Defaults to 1.
            decoder_heads (int, optional): Number of decoder heads. Defaults to 8.
            decoder_dim_head (int, optional): Size of each head. Defaults to 64.

        Returns:
            torch.Tensor: Mse loss between pixel values and decoder output.
        """
        super(MAE, self).__init__()
        assert masking_ratio > 0 and masking_ratio < 1
        self.masking_ratio = masking_ratio

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]


        self.enc_dec = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads,dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, x : torch.tensor) -> torch.tensor:
        """ Forward function for MAE

        Args:
            x (torch.tensor): Image tensor.

        Returns:
            torch.tensor: Loss value.
        """
        device = x.device

        patches = self.to_patch(x)
        batch, num_patches, _ = patches.shape

        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        masked_patches = patches[batch_range, masked_indices]

        encoded_tokens = self.encoder.transformer(tokens)

        decoder_tokens = self.enc_dec(encoded_tokens)
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim = 1)
        decoded_tokens = self.decoder(decoder_tokens)

        mask_tokens = decoded_tokens[:, :num_masked]
        pred_pixel_values = self.to_pixels(mask_tokens)


        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss