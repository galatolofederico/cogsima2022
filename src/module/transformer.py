import torch
from einops import rearrange
from x_transformers.x_transformers import AttentionLayers, AbsolutePositionalEmbedding

class Transfomer(torch.nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            causal=False,
            dropout=0,
            seq_len=None,
            pos_embedding=False,
            cross_attend=False
        ):
        super(Transfomer, self).__init__()

        self.attn_layers = AttentionLayers(
            dim = dim,
            depth = depth,
            heads = heads,
            cross_attend = cross_attend,
            causal = causal
        )

        self.pos_embedding = None
        if pos_embedding:
            assert seq_len is not None, "Must specify seq_len when using positional embeddings"
            self.pos_embedding = AbsolutePositionalEmbedding(dim, seq_len) 
        self.norm = torch.nn.LayerNorm(self.attn_layers.dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, return_attention=False, **kwargs):
        if self.pos_embedding is not None:
            embeddings = embeddings + self.pos_embedding(embeddings)
        
        embeddings = self.dropout(embeddings)
        latent, intermediates = self.attn_layers(embeddings, return_hiddens=True, **kwargs)
        latent = self.norm(latent)

        if return_attention:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return latent, attn_maps
        
        return latent


class ViT(torch.nn.Module):
    def __init__(self, 
        dim,
        depth,
        heads,
        image_size,
        patch_size,
        dropout = 0
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image dimensions must be divisible by the patch size"
        
        self.attn_layers = AttentionLayers(
            dim = dim,
            depth = depth,
            heads = heads,
            dropout = 0,
            causal = False
        )

        dim = self.attn_layers.dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size ** 2
        self.patch_size = patch_size

        self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = torch.nn.Linear(patch_dim, dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, img, return_attention=False, **kwargs):
        img = rearrange(img, "b (h p1) (w p2) -> b (h w) (p1 p2)", p1 = self.patch_size, p2 = self.patch_size)
        embeddings = self.patch_to_embedding(img)

        embeddings = embeddings + self.pos_embedding
        embeddings = self.dropout(embeddings)

        latent, intermediates = self.attn_layers(embeddings, return_hiddens=True, **kwargs)       
        latent = self.norm(latent)

        if return_attention:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return latent, attn_maps
        
        return latent