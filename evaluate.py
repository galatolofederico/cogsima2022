import argparse
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import sklearn.metrics
from tqdm import tqdm
import seaborn as sns

def regression_metrics(y_true, y_pred):
    return dict(
        mean_absolute_error = float(sklearn.metrics.mean_absolute_error(y_true, y_pred)),
        mean_squared_error = float(sklearn.metrics.mean_squared_error(y_true, y_pred)),
        d2_tweedie_score = float(sklearn.metrics.d2_tweedie_score(y_true, y_pred)),
        r2_score = float(sklearn.metrics.r2_score(y_true, y_pred)),
        explained_variance_score = float(sklearn.metrics.explained_variance_score(y_true, y_pred))
    )
    

def plot_run(outputs, folder):
    y_true = np.array(outputs["outputs"]["actuals"])
    y_pred = np.array(outputs["outputs"]["preds"])
    
    histograms = [
        (y_true, "values_hist.png"),
        (y_pred, "predictions_hist.png"),
        (y_true - y_pred, "errors_hist.png"),
        (np.abs(y_true - y_pred), "errors_abs_hist.png"),
    ]

    histograms_folder = os.path.join(folder, "histograms")
    os.makedirs(histograms_folder, exist_ok=True)
    for data, name in histograms:
        plt.figure()
        plt.hist(data, bins=30)
        plt.savefig(os.path.join(histograms_folder, name))
        plt.close()
    
    scatters = [
        (y_true, y_pred, "values_vs_predcition.png"),
        (y_true, np.abs(y_true - y_pred), "values_vs_abs_errors.png"),
        (y_pred, np.abs(y_true - y_pred), "predictions_vs_abs_errors.png"),
    ]

    scatters_folder = os.path.join(folder, "scatters")
    os.makedirs(scatters_folder, exist_ok=True)
    for x, y, name in scatters:
        plt.figure()
        plt.scatter(x, y, s=1, c="k")
        plt.savefig(os.path.join(scatters_folder, name))
        plt.close()


def plot_stochastic(outputs, folder):
    y_true = np.array(outputs["outputs"]["actuals"])
    y_pred = np.array(outputs["outputs"]["preds"])
    y_std = np.array(outputs["outputs"]["stds"])
    
    stochastic_folder = os.path.join(folder, "stochastic")
    os.makedirs(stochastic_folder, exist_ok=True)

    plt.figure()
    plt.hist(y_std, bins=30)
    plt.savefig(os.path.join(stochastic_folder, "std_hist.png"))
    plt.close()

    percs = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]
    stochastic = []
    for perc in percs:
        upper = y_pred + perc*y_std
        bottom  = y_pred - perc*y_std

        rights = ((y_pred > bottom) & (y_true < upper)).astype(float).sum()
        accuracy = rights / len(y_pred)
        
        stochastic.append(dict(
            std_perc=perc,
            accuracy=accuracy
        ))

    stochastic_df = pd.DataFrame(stochastic)
    
    stochastic_df.to_csv(os.path.join(stochastic_folder, "stochastic_accuracy.csv"))

    plt.figure()
    stochastic_df["accuracy"].plot.line()
    plt.show()


def evaluate(args):
    overall_folder = os.path.join(args.results_folder, "overall")
    metrics_folder = os.path.join(overall_folder, "metrics")
    plots_folder = os.path.join(overall_folder, "plots")
    
    os.makedirs(metrics_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    
    metrics = []
    for folder in tqdm(os.listdir(args.results_folder)):
        folder = os.path.join(args.results_folder, folder)
        if not "outputs.json" in os.listdir(folder):
            continue
        with open(os.path.join(folder, "outputs.json"), "r") as f:
            outputs = json.load(f)
        
        run_metrics = regression_metrics(
            outputs["outputs"]["actuals"],
            outputs["outputs"]["preds"]
        )
        
        data = outputs["metadata"]
        data.update(run_metrics)
        metrics.append(data)

        plot_run(outputs, folder)
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(metrics_folder, "metrics.csv"))

    metrics_names = list(metrics_df.columns)
    metrics_names.remove("points") 
    metrics_names.remove("model")

    for metric in tqdm(metrics_names):
        plt.figure()
        sns.lineplot(data=metrics_df, x="points", y=metric, hue="model")
        plt.savefig(os.path.join(plots_folder, f"{metric}.png"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-folder", type=str, default="./results/test.json")
    args = parser.parse_args()

    evaluate(args)