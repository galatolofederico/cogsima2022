import argparse
import os
import random
import json

parser = argparse.ArgumentParser()

parser.add_argument("--data-folders", nargs="+", required=True)
parser.add_argument("--output-folder", type=str, default="./dataset")
parser.add_argument("--train-perc", type=float, default=0.8)

args = parser.parse_args()

dataset = []
for data_dir in args.data_folders:
    dirs = os.listdir(data_dir)
    dirs = list(filter(lambda d: os.path.isdir(os.path.join(data_dir, d)), dirs))
    assert "geo" in dirs
    dirs.remove("geo")

    dataset.append(dict(
        dirs=[os.path.join(data_dir, d) for d in dirs],
        geo=os.path.join(data_dir, "geo")
    ))

train_dataset = []
test_dataset = []

for elem in dataset:
    dirs = elem["dirs"]
    random.shuffle(dirs)

    train_pivot = int(len(dirs)*args.train_perc)
    train = dirs[:train_pivot]
    test = dirs[train_pivot:]

    train_dataset.append(dict(
        dirs=train,
        geo=elem["geo"]
    ))
    test_dataset.append(dict(
        dirs=test,
        geo=elem["geo"]
    ))

print(f"Train: {sum([len(d['dirs']) for d in train_dataset])}")
print(f"Test: {sum([len(d['dirs']) for d in test_dataset])}")

with open(os.path.join(args.output_folder, "train.json"), "w") as f:
    json.dump(train_dataset, f)
with open(os.path.join(args.output_folder, "test.json"), "w") as f:
    json.dump(test_dataset, f)
with open(os.path.join(args.output_folder, "dev.json"), "w") as f:
    json.dump([
        dict(
            dirs=train_dataset[0]["dirs"][:2],
            geo=train_dataset[0]["geo"]
        )
    ], f)
