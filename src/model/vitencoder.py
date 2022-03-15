import torch
import torch.nn.functional as F
from einops import rearrange

from src.module.transformer import Transfomer, ViT
from src.module.mlm import RasterMLM

class ViTEncoderTransformer(RasterMLM):
    def __init__(self, config):
        self.config = config
        self.model_config = self.config.model.transformer.vitencoder

        super(ViTEncoderTransformer, self).__init__(
            embeddings=self.config.dataset.raster.sample.size,
            dim=self.model_config.dim
        )

        self.dem_encoder = ViT(
            dim = self.model_config.dim,
            depth = self.model_config.depth,
            heads = self.model_config.heads,
            image_size = self.config.dataset.dem.size, 
            patch_size = self.model_config.patch_size
        )

        self.data_encoder = Transfomer(
            dim = self.model_config.dim,
            depth = self.model_config.depth,
            heads = self.model_config.heads,
            seq_len = self.config.dataset.raster.sample.max_points*2,
            pos_embedding =  True,
            cross_attend = True,
            causal = False
        )

        self.regression_head = torch.nn.Linear(
            self.model_config.dim,
            1
        )

    @property
    def name(self):
        return "vitencoder"

    def forward(self, ij_embeddings, v_embeddings, batch, return_attention=False):
        data_embeddings = torch.cat((ij_embeddings, v_embeddings), dim=1)

        dem_encodings, dem_attention = self.dem_encoder(
            batch["dem"],
            return_attention=True
        )
        data_encodings, data_attention = self.data_encoder(
            data_embeddings,
            context=dem_encodings,
            return_attention=True
        )
        assert data_encodings.shape[1] % 2 == 0
        values_encodings = data_encodings[:, data_encodings.shape[1]//2:, :]

        if return_attention:
            return self.regression_head(values_encodings)[:, :, 0], [
                dem_attention,
                data_attention
            ]
        else:
            return self.regression_head(values_encodings)[:, :, 0]