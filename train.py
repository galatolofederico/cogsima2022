import argparse
import os
from torch.utils.data import DataLoader
import wandb
import torch
from tqdm import tqdm

from src.dataset.raster import RasterDataset

from src.model.encoderencoder import EncoderEncoderTransformer
from src.model.vitencoder import ViTEncoderTransformer
from src.model.encoderdecoder import EncoderDecoderTransformer
from src.model.vitdecoder import ViTDecoderTransformer

from src.utils import load_config, deep_move

def train(config):
    dataset = RasterDataset(
        config.args.dataset,
        config.dataset,
        return_dem=True,
        return_tokens=True,
        return_values=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        num_workers=1    
    )

    if config.args.model == "encoderencoder":
        Model = EncoderEncoderTransformer
    elif config.args.model == "vitencoder":
        Model = ViTEncoderTransformer
    elif config.args.model == "encoderdecoder":
        Model = EncoderDecoderTransformer
    elif config.args.model == "vitdecoder":
        Model = ViTDecoderTransformer
    else:
        raise Exception(f"Unknown model {config.args.model}")
    

    model = Model(config).to(config.args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
    
    if config.args.wandb:
        wandb.init(project=config.args.wandb_project, entity=config.args.wandb_entity)
    
    if config.args.train_batches > 0:
        pbar = tqdm(zip(range(0, config.args.train_batches+1), dataloader), total=config.args.train_batches+1)
    else:
        pbar = tqdm(enumerate(dataloader))
    for i, batch in pbar:
        batch = deep_move(batch, config.args.device)
        outs = model.do_train(batch)
        
        loss = outs["loss"]

        pbar.set_description(f"loss: {loss.item():.2f}")
        if config.args.wandb:
            wandb.log(dict(loss=loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if config.args.save and i > 0 and i % config.train.checkpoint_every == 0:
            os.makedirs(os.path.dirname(config.model.transformer.checkpoint), exist_ok=True)
            torch.save(model, config.model.transformer.checkpoint.format(model=config.args.model))
            print()
            print("Model Saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--train-batches", type=int, default=1000)
    parser.add_argument("--dataset", type=str, default="./dataset/train.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default="mlpi")
    parser.add_argument("--wandb-project", type=str, default="subsidence")
    parser.add_argument("--wandb-tag", type=str, default="")

    args = parser.parse_args()
    config = load_config(args)

    train(config)
