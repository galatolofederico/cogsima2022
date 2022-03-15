import argparse
import random
import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import geopandas as gpd

from src.utils import deep_move

parser = argparse.ArgumentParser()

parser.add_argument("--raster", type=str, required=True)
parser.add_argument("--geo", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--min-points", type=int, default=25)
parser.add_argument("--max-points", type=int, default=300)
parser.add_argument("--montecarlo-steps", type=int, default=10000)
parser.add_argument("--sample-size", type=int, default=20)
parser.add_argument("--max-predictions", type=int, default=100)

args = parser.parse_args()

assert args.sample_size % 2 == 0
sample_offset = int(args.sample_size / 2)

model = torch.load(args.model).to(args.device)

with open(os.path.join(args.geo, "lon.npy"), "rb") as f:
    lon = np.load(f)
with open(os.path.join(args.geo, "lat.npy"), "rb") as f:
    lat = np.load(f)
with open(os.path.join(args.raster, "count.npy"), "rb") as f:
    count = np.load(f)
with open(os.path.join(args.raster, "raster.npy"), "rb") as f:
    raster = np.load(f)

zones = np.argwhere(count != 0)
predicted_raster  = np.zeros_like(raster)
predicted_count  = np.zeros_like(count)

for _ in tqdm(range(0, args.montecarlo_steps)):
    center_i = random.randint(0, raster.shape[0]-1)
    center_j = random.randint(0, raster.shape[1]-1)


    if center_i-sample_offset < 0 or \
    center_i+sample_offset >= raster.shape[0] or \
    center_j-sample_offset < 0 or \
    center_j+sample_offset >= raster.shape[1]:
        continue

    data_patch = raster[
        center_i-sample_offset:center_i+sample_offset,
        center_j-sample_offset:center_j+sample_offset
    ]
    count_patch = count[
        center_i-sample_offset:center_i+sample_offset,
        center_j-sample_offset:center_j+sample_offset
    ]

    data_points = np.argwhere(count_patch > 0)
    target_points = np.argwhere(count_patch == 0)

    data_points = data_points[np.random.permutation(len(data_points))]
    target_points = target_points[np.random.permutation(len(target_points))]
    target_point = target_points[0]

    if len(data_points) < args.min_points:
            continue

    target_i = center_i + target_point[0] - sample_offset
    target_j = center_j + target_point[1] - sample_offset

    if count[target_i, target_j] > 0:
        continue


    sample_points = data_points[:args.max_points-1]
    sample_points = np.concatenate((
        sample_points,
        np.atleast_2d(target_point)
    ))
    sample_values = data_patch[sample_points[:, 0], sample_points[:, 1]]

    values = sample_values.flatten()
    i_tokens  = sample_points[:, 0].flatten()
    j_tokens  = sample_points[:, 1].flatten()

    batch = dict(
        values=torch.tensor(values).float().unsqueeze(0),
        tokens=dict(
            i=torch.tensor(i_tokens).float().unsqueeze(0),
            j=torch.tensor(j_tokens).float().unsqueeze(0)
        )
    )

    prediction = model.do_predict(deep_move(batch, args.device))["regression"].item()

    predicted_raster[target_i, target_j] += prediction
    predicted_count[target_i, target_j] += 1


predicted_raster = predicted_raster / predicted_count

df = list()
for i in range(0, raster.shape[0]):
    for j in range(0, raster.shape[1]):
        current_value = raster[i, j]
        current_count = count[i, j]

        current_predicted_value = predicted_raster[i, j]
        current_predicted_count = predicted_count[i, j]
        
        current_lon = lon[i, j]
        current_lat = lat[i, j]

        ret = dict(
            Longitude=current_lat,
            Latitude=current_lon,
        )

        assert not (current_count > 0 and current_predicted_count > 0)

        add_point = False
        if current_count > 0:
            ret["input_value"] = current_value
            ret["value"] = current_value
            add_point = True
        if current_predicted_count > 0:
            ret["predicted_value"] = current_predicted_value
            ret["value"] = current_predicted_value
            add_point = True
        
        if add_point: df.append(ret)

df = pd.DataFrame(df)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
gdf.to_file(args.output)