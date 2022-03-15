import torch
import os
import numpy as np
import random
import rasterio as rio
import skimage.transform
import fiona.transform

class DemDataset:
    def __init__(self, config):
        self.config = config

        self.dem_dataset = rio.open(self.config.dem.file)
        assert self.dem_dataset.count == 1, "DEM should have 1 channel"

    def from_patch(self, patch):
        xx, yy = fiona.transform.transform(
            self.config.raster.crs,
            self.dem_dataset.crs.to_proj4(),
            [
                patch["geo"]["left_bot"][0],
                patch["geo"]["right_top"][0]
            ],
            [
                patch["geo"]["left_bot"][1],
                patch["geo"]["right_top"][1]
            ],
        )
        
        left_bot = self.dem_dataset.index(xx[0], yy[0])
        right_top = self.dem_dataset.index(xx[1], yy[1])

        window = rio.windows.Window.from_slices(
            (right_top[0], left_bot[0]),
            (left_bot[1], right_top[1])
        )
        
        dem = self.dem_dataset.read(window=window)[0]
        dem[dem < 0] = 0

        dem = skimage.transform.resize(dem, (self.config.dem.size, self.config.dem.size))
        
        if dem.std() == 0 or np.isinf(dem.std()) or np.isnan(dem.std()):
            return None

        dem = (dem - dem.mean()) / dem.std()

        return dem
