import torch
import os
import json
import numpy as np
import random
import joblib

from src.dataset.dem import DemDataset


def padarray(arr, size, value):
    assert arr.shape[0] <= size
    if arr.shape[0] == size: return arr
    ret = np.full((size, *arr.shape[1:]), value)
    ret[:arr.shape[0]] = arr
    return ret


class RasterDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            dataset,
            config,
            return_data_key=False,
            return_center_i=False,
            return_patch=False,
            return_values=False,
            return_dem=False,
            return_tokens=True,
            return_all=False,
            seed=None
        ):
        self.config = config
        self.return_data_key = return_data_key
        self.return_center_i = return_center_i
        self.return_patch = return_patch
        self.return_values = return_values
        self.return_dem = return_dem
        self.return_tokens = return_tokens
        self.return_all = return_all
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        assert self.config.raster.sample.size % 2 == 0
        self.sample_offset = int(self.config.raster.sample.size/2)

        self.dem_dataset = DemDataset(self.config)
        
        with open(dataset, "r") as f:
            self.dataset = json.load(f)

        self.data = dict()
        for elem in self.dataset:
            geo = dict()
            with open(os.path.join(elem["geo"], "lon.npy"), "rb") as f:
                geo["lon"] = np.load(f)
            with open(os.path.join(elem["geo"], "lat.npy"), "rb") as f:
                geo["lat"] = np.load(f)
            
            for subdir in elem["dirs"]:
                files = os.listdir(subdir)
                self.data[subdir] = dict(
                    geo=geo,
                )
                for file in filter(lambda f: f.endswith(".npy"), files):
                    with open(os.path.join(subdir, file), "rb") as f:
                        self.data[subdir][file.split(".")[0]] = np.load(f)

        for key, data in self.data.items():
            self.data[key]["zones"] = np.argwhere(data["count"] != 0)

        self.data_keys = list(self.data.keys())

    def extract_patch(self, data, center):
        if center[0]-self.sample_offset < 0 or \
        center[0]+self.sample_offset >= data["raster"].shape[0] or \
        center[1]-self.sample_offset < 0 or \
        center[1]+self.sample_offset >= data["raster"].shape[1]:
            return None

        data_patch = data["raster"][
            center[0]-self.sample_offset:center[0]+self.sample_offset,
            center[1]-self.sample_offset:center[1]+self.sample_offset
        ]
        count_patch = data["count"][
            center[0]-self.sample_offset:center[0]+self.sample_offset,
            center[1]-self.sample_offset:center[1]+self.sample_offset
        ]

        lon_patch = data["geo"]["lat"][
            center[0]-self.sample_offset:center[0]+self.sample_offset,
            center[1]-self.sample_offset:center[1]+self.sample_offset
        ]
        lat_patch = data["geo"]["lon"][
            center[0]-self.sample_offset:center[0]+self.sample_offset,
            center[1]-self.sample_offset:center[1]+self.sample_offset
        ]

        left_bot  = lon_patch.min(), lat_patch.min()
        right_top = lon_patch.max(), lat_patch.max()

        query  = f'"Longitude" > {left_bot[0]}  AND '
        query += f'"Longitude" < {right_top[0]} AND '
        query += f'"Latitude"  > {left_bot[1]}  AND '
        query += f'"Latitude"  < {right_top[1]}'

        return dict(
            data = data_patch,
            count = count_patch,
            geo = dict(
                left_bot = left_bot,
                right_top = right_top,
                query = query
            )
        )

    def sample_points(self, patch):
        data_points = np.argwhere(patch["count"] > 0)

        if len(data_points) < self.config.raster.sample.min_points:
            return None
        
        sample_points = data_points[np.random.permutation(len(data_points))][:self.config.raster.sample.max_points]
        sample_values = patch["data"][sample_points[:, 0], sample_points[:, 1]]

        values = padarray(
            arr=sample_values.flatten(),
            size=self.config.raster.sample.max_points,
            value=float(self.config.raster.sample.pad_value),
        )

        return dict(
            points = sample_points,
            values = torch.tensor(values).float()
        )
    
    def tokenize(self, patch, points):
        i_tokens  = points["points"][:, 0].flatten()
        j_tokens  = points["points"][:, 1].flatten()
        ij_tokens = i_tokens*patch["data"].shape[0] + j_tokens

        
        i_tokens  = padarray(
            arr=i_tokens,
            size=self.config.raster.sample.max_points,
            value=float(self.config.raster.sample.pad_value),
        )
        j_tokens  = padarray(
            arr=j_tokens,
            size=self.config.raster.sample.max_points,
            value=float(self.config.raster.sample.pad_value),
        )
        ij_tokens = padarray(
            arr=ij_tokens,
            size=self.config.raster.sample.max_points,
            value=float(self.config.raster.sample.pad_value),
        )

        return dict(
            tokens=dict(
                i=torch.tensor(i_tokens).long(),
                j=torch.tensor(j_tokens).long(),
                ij=torch.tensor(ij_tokens).long(),
            )
        )


    def __iter__(self):
        return self

    def __next__(self):
        while True:
            data_key = random.choice(self.data_keys)
            data = self.data[data_key]
            center_i = random.randint(0, len(data["zones"])-1)
            center = data["zones"][center_i]

            patch = self.extract_patch(data, center)
            if patch is None: continue
            
            points = self.sample_points(patch)
            if points is None: continue

            dem = self.dem_dataset.from_patch(patch)
            if dem is None: continue
            tokens = self.tokenize(patch, points)

            ret = dict()

            if self.return_data_key or self.return_all: ret["data_key"] = data_key
            if self.return_center_i or self.return_all: ret["center_i"] = center_i
            if self.return_patch or self.return_all: ret["patch"] = patch
            if self.return_values or self.return_all: ret["values"] = points["values"]
            if self.return_dem or self.return_all: ret["dem"] = dem            
            if self.return_tokens or self.return_all: ret["tokens"] = tokens["tokens"]

            return ret

if __name__ == "__main__":
    from src.utils import load_config
    from matplotlib import pyplot as plt
    import numpy as np

    config = load_config()

    ds = RasterDataset("./dataset/train.json", config.dataset, return_all=True, seed=42)
    
    for e in ds:
        fig, axes = plt.subplots(2)
        print("datakey", e["data_key"])
        print("geo", e["patch"]["geo"])
        print("dem.shape", e["dem"].shape)
        print("dem min/max", e["dem"].min(), e["dem"].max())

        axes[0].matshow(e["patch"]["data"], cmap="Reds")
        axes[1].matshow(e["dem"])

        plt.show()
