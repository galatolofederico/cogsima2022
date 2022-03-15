import torch
import yaml
from addict import Dict

def load_config(args = None):
    with open(args.config if args is not None else "./config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    if args is not None:
        config["args"] = vars(args)

    return Dict(config)


def deep_move(elem, device):
    if isinstance(elem, torch.Tensor):
        return elem.to(device)
    elif isinstance(elem, dict):
        for k, v in elem.items():
            elem[k] = deep_move(v, device)
    elif isinstance(elem, list):
        for i, e in enumerate(elem):
            elem[i] = deep_move(e, device)
    else:
        return elem
    return elem