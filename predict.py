import torch
import numpy as np
import argparse
import os
import sys
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

import sklearn.metrics
from sklearn.neighbors import KNeighborsRegressor
from src.utils import load_config, deep_move
from src.dataset.raster import RasterDataset

def regression_metrics(y_true, y_pred):
    return dict(
        mean_absolute_error = float(sklearn.metrics.mean_absolute_error(y_true, y_pred)),
        mean_squared_error = float(sklearn.metrics.mean_squared_error(y_true, y_pred)),
        d2_tweedie_score = float(sklearn.metrics.d2_tweedie_score(y_true, y_pred)),
        r2_score = float(sklearn.metrics.r2_score(y_true, y_pred)),
        explained_variance_score = float(sklearn.metrics.explained_variance_score(y_true, y_pred))
    )
    

class TransformerModel:
    def __init__(self, config):
        self.config = config
        self.model = torch.load(config.args.model).to(config.args.device)
        print(f"Loaded {config.args.model}")
        self.model.eval()

    @property
    def name(self):
        return f"transformer-{self.model.name}"

    def predict(self, batch):
        batch = deep_move(batch, self.config.args.device)

        with torch.no_grad():
            outs = self.model.do_predict(batch)

        ret = dict()
        ret["regression"] = outs["regression"].cpu().numpy()
        
        return ret

class KNNModel:
    def __init__(self, config):
        self.config = config
        self.n_neighbors = int(self.config.args.points*self.config.args.knn_perc)
    
    @property
    def name(self):
        return f"knn"
    
    def predict(self, batch):
        knn = KNeighborsRegressor(n_neighbors=self.n_neighbors)

        i_tokens = batch["tokens"]["i"].float().cpu().numpy()
        j_tokens = batch["tokens"]["j"].float().cpu().numpy()
        values = batch["values"].cpu().numpy()

        i_target = i_tokens[:, -1]
        j_target = j_tokens[:, -1]

        i_input = i_tokens[:, :-1].flatten()
        j_input = j_tokens[:, :-1].flatten()

        X = np.vstack((i_input, j_input)).T
        y = values[:, :-1].flatten()

        knn.fit(X, y)

        X_target = np.vstack((i_target, j_target)).T
        y_pred = knn.predict(X_target)
        
        return dict(
            regression=y_pred
        )


def predict(model, dataloader, config, eval_batches=100):
    regression_outputs = dict(
        actuals=list(),
        preds=list(),
    )

    print("Predicting...")
    for i, batch in tqdm(zip(range(0, eval_batches), dataloader), total=eval_batches):
        right_values = batch["values"][:, -1].clone().cpu().numpy()
        batch["values"][:, -1] = float("nan")

        outs = model.predict(batch)

        regression_outputs["actuals"].append(right_values)
        regression_outputs["preds"].append(outs["regression"])
    
    regression_outputs["actuals"] = np.concatenate(regression_outputs["actuals"]).tolist()
    regression_outputs["preds"] = np.concatenate(regression_outputs["preds"]).tolist()
  
    return regression_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--knn-perc", type=int, default=0.25)
    parser.add_argument("--dataset", type=str, default="./dataset/test.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-folder", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--justmetrics", action="store_true")
    
    args = parser.parse_args()

    config = load_config(args)
    config.dataset.raster.sample.min_points = config.args.points
    config.dataset.raster.sample.max_points = config.args.points

    os.makedirs(config.args.output_folder, exist_ok=True)
    
    torch.manual_seed(config.args.seed)
    dataset = RasterDataset(
        config.args.dataset,
        config.dataset,
        return_dem=True,
        return_tokens=True,
        return_values=True,
        return_data_key=True,
        seed=config.args.seed
    )
    dataloader = DataLoader(dataset, batch_size=16, num_workers=1, shuffle=False)
    
    if os.path.exists(config.args.model):
        model = TransformerModel(config)
    elif config.args.model == "knn":
        model = KNNModel(config)

    outputs = predict(model, dataloader, config, args.eval_batches)
    metadata = dict(
        points=config.args.points,
        model=model.name
    )

    if args.justmetrics:
        results = regression_metrics(
            outputs["actuals"],
            outputs["preds"],
        )
        print(json.dumps(results, indent=4))
        sys.exit(0)

    dataset_name = os.path.basename(config.args.dataset)
    model_fullname = f"{model.name}-{config.args.points}"
    output_folder = os.path.join(config.args.output_folder, dataset_name, model_fullname)
    os.makedirs(output_folder, exist_ok=True)
    fname = os.path.join(output_folder, "outputs.json")
    with open(fname, "w") as f:
        json.dump(dict(
            outputs=outputs,
            metadata=metadata
        ), f)
    print(f"Results written to: {fname}")


