import torch
import torch.nn.functional as F
from einops import rearrange

from src.module.embedder import RasterEmbedder

class RasterAutoRegressive(RasterEmbedder):
    def __init__(self, embeddings, dim):
        super(RasterAutoRegressive, self).__init__(embeddings, dim)
        assert hasattr(self, "config")

    def do_train(self, batch, return_attention=False):
        ij_embeddings, v_embeddings, pad_mask = self.embed(batch)
        
        in_v_embeddings = v_embeddings[:, :-1, :]
        out_values = batch["values"][:, 1:]
        
        regression = self(ij_embeddings, in_v_embeddings, batch)
        
        return dict(loss=F.l1_loss(regression, out_values))
        
    def do_predict(self, batch, return_attention=False):
        ij_embeddings, v_embeddings, pad_mask = self.embed(batch)

        in_v_embeddings = v_embeddings[:, :-1, :]
        
        regression, attention = self(ij_embeddings, in_v_embeddings, batch, return_attention=True)
        
        ret = dict(regression=regression[:, -1])
        if return_attention: ret["attention"] = attention

        return ret