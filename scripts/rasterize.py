import argparse
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import math
from tqdm import tqdm
from matplotlib import pyplot as plt

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6_371_000

    return c * r

parser = argparse.ArgumentParser()

parser.add_argument("--shp", type=str, required=True)
parser.add_argument("--output-folder", type=str, default="./dataset/raster")
parser.add_argument("--field-column", type=str, default="field_")
parser.add_argument("--raster-size", type=int, default=100)
parser.add_argument("--save-shapefile", action="store_true")

args = parser.parse_args()

shp = gpd.read_file(args.shp)
print(f"Loaded {len(shp)} rows")
field_columns = list(filter(lambda col: col.startswith(args.field_column), shp.columns))
print(f"Using {len(field_columns)} fields: {field_columns}")

left_bot  = shp["Lon"].min(), shp["Lat"].min()
right_top = shp["Lon"].max(), shp["Lat"].max()

print(f"Left bottom: {left_bot}")
print(f"Right top: {right_top}")

hdist = haversine(left_bot[0], left_bot[1], right_top[0], left_bot[1])
vdist = haversine(left_bot[0], left_bot[1], left_bot[0], right_top[1])

print(f"Horizontal distance: {hdist}")
print(f"Vertical distance: {hdist}")

hnum = math.ceil(hdist/args.raster_size)
vnum = math.ceil(vdist/args.raster_size)

print(f"Horizontal cells: {hnum}")
print(f"Vertical cells: {vnum}")

harc = (right_top[0] - left_bot[0])/hnum
varc = (right_top[1] - left_bot[1])/vnum

print(f"Horizontal cell arc: {harc}")
print(f"Vertical cell arc: {varc}")

raster = dict()
for field in field_columns:
    raster[field] = dict(
        raster = np.zeros((hnum+1, vnum+1)),
        count  = np.zeros((hnum+1, vnum+1)),
    )

raster["lon"] = np.zeros((hnum+1, vnum+1))
raster["lat"] = np.zeros((hnum+1, vnum+1))

print("Processing...")
for idx, elem in tqdm(shp.iterrows(), total=len(shp)):
    i_perc = (elem["Lon"] - left_bot[0])/(right_top[0] - left_bot[0])
    j_perc = (elem["Lat"] - left_bot[1])/(right_top[1] - left_bot[1])
    
    i = int(i_perc*hnum)
    j = int(j_perc*vnum)
    
    for field in field_columns:
        raster[field]["raster"][i, j] += elem[field]
        raster[field]["count"][i, j]  += 1
    
print("Referencing...")
for i in tqdm(range(0, hnum+1)):
    for j in range(0, vnum+1):
        raster_lon = left_bot[1] + (j + 1/2)*varc
        raster_lat = left_bot[0] + (i + 1/2)*harc
        
        raster["lon"][i, j] = raster_lon
        raster["lat"][i, j] = raster_lat
        

name = os.path.basename(args.shp).split(".")[0]
output_folder = os.path.join(args.output_folder, name)
os.makedirs(output_folder, exist_ok=True)

print("Saving...")

geo_folder = os.path.join(output_folder, "geo")
os.makedirs(geo_folder, exist_ok=True)

with open(os.path.join(geo_folder, "lon.npy"), "wb") as f:
    np.save(f, raster["lon"])
with open(os.path.join(geo_folder, "lat.npy"), "wb") as f:
    np.save(f, raster["lat"])


for field in tqdm(field_columns):
    raster[field]["raster"] = np.divide(raster[field]["raster"], raster[field]["count"], out=np.zeros_like(raster[field]["raster"]), where=(raster[field]["count"] != 0))
    
    dest = os.path.join(output_folder, field)
    os.makedirs(dest, exist_ok=True)
    
    for channel in raster[field]:
        with open(os.path.join(dest, f"{channel}.npy"), "wb") as f:
            np.save(f, raster[field][channel])
        
    if args.save_shapefile:
        df = list()
        for i in range(0, raster[field]["raster"].shape[0]):
            for j in range(0, raster[field]["raster"].shape[1]):
                value = raster[field]["raster"][i, j]
                count = raster[field]["count"][i, j]
                
                lon = raster["lon"][i, j]
                lat = raster["lat"][i, j]

                if count > 0:        
                    df.append(dict(
                        value=value,
                        Longitude=lat,
                        Latitude=lon,
                    ))
            
        df = pd.DataFrame(df)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
        gdf.to_file(os.path.join(dest, f"raster.shp"))