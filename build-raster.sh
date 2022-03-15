#!/bin/sh

python -m scripts.rasterize --shp ./dataset/shp/bologna-asc/bologna-asc.shp --output-folder ./dataset/raster/
python -m scripts.rasterize --shp ./dataset/shp/bologna-dsc/bologna-dsc.shp --output-folder ./dataset/raster/
python -m scripts.rasterize --shp ./dataset/shp/pistoia-asc/pistoia-asc.shp --output-folder ./dataset/raster/
python -m scripts.rasterize --shp ./dataset/shp/pistoia-dsc/pistoia-dsc.shp --output-folder ./dataset/raster/
