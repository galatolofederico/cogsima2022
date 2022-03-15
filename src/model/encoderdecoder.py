import torch
import torch.nn.functional as F
from einops import rearrange

from src.module.transformer import Transfomer
from src.module.autoregressive import RasterAutoRegressive

class EncoderDecoderTransformer(RasterAutoRegressive):
    def __init__(self, config):
        self.config = config
        self.model_config = self.config.model.transformer.encoderdecoder
        super(EncoderDecoderTransformer, self).__init__(
            embeddings=self.config.dataset.raster.sample.size,
            dim=self.model_config.dim
        )

        self.coordinates_encoder = Transfomer(
            dim = self.model_config.dim,
            depth = self.model_config.depth,
            heads = self.model_config.heads,
            seq_len = self.config.dataset.raster.sample.max_points,
            pos_embedding =  True,
            cross_attend = False,
            causal = False
        )

        self.values_decoder = Transfomer(
            dim = self.model_config.dim,
            depth = self.model_config.depth,
            heads = self.model_config.heads,
            seq_len = self.config.dataset.raster.sample.max_points,
            pos_embedding =  True,
            cross_attend = True,
            causal = True
        )

        self.regression_head = torch.nn.Linear(
            self.model_config.dim,
            1
        )

    @property
    def name(self):
        return "encoderdecoder"

    def forward(self, ij_embeddings, v_embeddings, batch, return_attention=False):
        ij_encodings, ij_attention = self.coordinates_encoder(
            ij_embeddings,
            return_attention=True
        )
        values_encodings, values_attention = self.values_decoder(
            v_embeddings,
            context=ij_encodings,
            return_attention=True
        )

        if return_attention:
            return self.regression_head(values_encodings)[:, :, 0], [
                ij_attention,
                values_attention
            ]
        else:
            return self.regression_head(values_encodings)[:, :, 0]
   