import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import DSTI_HangZhou
from dataset_hangzhou import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/hangzhou_final_25" + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, train_scaler, valid_scaler, test_scaler = get_dataloader(
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    missing_pattern= config["model"]["missing_pattern"],
    device = args.device,
    missing_ratio=config["model"]["test_missing_ratio"],
)

adj = torch.load('A_hat.pt').to(args.device)
adjm = torch.load('A_hat.pt').to(args.device)
print(adj.shape)
model = DSTI_HangZhou(config, args.device).to(args.device)
if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        adj, adjm,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, test_loader, adj, adjm, nsample=args.nsample, scaler=1,
    mean_scaler=0, test_scaler = test_scaler, foldername=foldername)


