import torch
import torch.nn.functional as F
from einops import rearrange

from src.module.embedder import RasterEmbedder

class RasterMLM(RasterEmbedder):
    def __init__(self, embeddings, dim):
        super(RasterMLM, self).__init__(embeddings, dim)
        assert hasattr(self, "config")

    def do_train(self, batch, return_attention=False):
        ij_embeddings, v_embeddings, pad_mask = self.embed(batch)

        probs = torch.zeros(*v_embeddings.shape[:2]).float().uniform_(0, 1).to(v_embeddings.device)
        mask = probs < self.config.model.transformer.mlm.mask_prob
        mask = mask & ~pad_mask
        target_mask = probs < (self.config.model.transformer.mlm.mask_prob + self.config.model.transformer.mlm.include_prob)
        target_mask = target_mask & ~pad_mask

        v_embeddings[mask] = self.mask_embedding
        
        regression = self(ij_embeddings, v_embeddings, batch)

        mask_values = batch["values"][target_mask]
        mask_regression = regression[target_mask]
        
        return dict(loss=F.l1_loss(mask_regression, mask_values))
        
    def do_predict(self, batch, return_attention=False):
        ij_embeddings, v_embeddings, pad_mask = self.embed(batch)

        mask = torch.zeros(*v_embeddings.shape[:2]).float()
        mask[:, -1] = 1
        mask = mask.bool().to(v_embeddings.device)

        v_embeddings[mask] = self.mask_embedding
        regression, attention = self(ij_embeddings, v_embeddings, batch, return_attention=True)

        ret = dict(regression=regression[mask])
        if return_attention: ret["attention"] = attention
        
        return ret