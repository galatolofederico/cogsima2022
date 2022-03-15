#!/bin/sh

usage() { echo "Usage: $0 -m <model> -s <input-shapefile> -f <field_name> -o <output-shapefile> -n <montecarlo-steps (default 10000)>" 1>&2; exit 1; }

while getopts ":s:f:m:o:n:" a; do
    case "${a}" in
        m)
            m=${OPTARG}
            ;;
        s)
            s=${OPTARG}
            ;;
        f)
            f=${OPTARG}
            ;;
        o)
            o=${OPTARG}
            ;;
        n)
            n=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${m}" ] || [ -z "${s}" ] || [ -z "${f}" ] || [ -z "${o}" ]; then
    usage
fi

if [ -z "${n}" ]; then
    n=10000
fi

out_folder="$(basename "$s" .shp)"

mkdir -p "$(dirname $o)"
python -m scripts.rasterize --shp "$s" --output-folder /tmp/predict-fill-shp
python predict-fill.py --raster "/tmp/predict-fill-shp/$out_folder/$f" --geo "/tmp/predict-fill-shp/$out_folder/geo" --model "$m" --output "$o" --montecarlo-steps "$n"
