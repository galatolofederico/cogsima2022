import torch
import torch.nn.functional as F
from einops import rearrange

class RasterEmbedder(torch.nn.Module):
    def __init__(self, embeddings, dim):
        super(RasterEmbedder, self).__init__()
        assert hasattr(self, "config")

        self.ij_embedding = torch.nn.Linear(2, dim)
        self.v_embedding = torch.nn.Linear(1, dim)

        self.mask_embedding = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.ij_pad_embedding = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.v_pad_embedding = torch.nn.Parameter(torch.randn(1, 1, dim))

    def embed(self, batch):
        i_tokens = rearrange(batch["tokens"]["i"], "b n -> b n ()")
        j_tokens = rearrange(batch["tokens"]["j"], "b n -> b n ()")
        values = rearrange(batch["values"], "b n -> b n ()")
        ij_tokens = torch.cat((i_tokens, j_tokens), dim=2).float()
        
        pad_mask = (i_tokens == self.config.dataset.raster.sample.pad_value)[:, :, 0]
        
        # placeholders to exclude from CG 
        ij_tokens[pad_mask] = 0
        values[pad_mask] = 0

        # compute embeddings
        ij_embeddings = self.ij_embedding(ij_tokens)
        v_embeddings = self.v_embedding(values)

        # replace placeholders with masks
        ij_embeddings[pad_mask] = self.ij_pad_embedding
        v_embeddings[pad_mask] = self.v_pad_embedding

        return ij_embeddings, v_embeddings, pad_mask
