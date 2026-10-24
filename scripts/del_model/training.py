import pandas as pd
import yaml
from tqdm import tqdm
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import os

import sys
sys.path.append(".")

from delta_del.dataloader import get_dataloader
from delta_del.helpers import set_seed
from delta_del.models import get_model

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--split-type", type=str, default="random", choices=["random", "disynthon"])
    parser.add_argument("--target-id", type=int, required=True, choices=[0, 1], help="Target ID: 0 for DDR1, 1 for MAPK14")

    return parser.parse_args()


def sklearn_concat(loader):
    X_list, y_list = [], []

    for batch in loader:
        X_list.append(batch["ecfp"].numpy())
        y_list.append(batch["enrichment"].numpy().astype(np.float32))

    return np.vstack(X_list), np.concatenate(y_list)


# ----------------------------
# MAIN
# ----------------------------
args = parse_args()
set_seed(123)


split = args.split
split_type = args.split_type
target_id = args.target_id

if target_id == 0:
    target_name = "ddr1"
elif target_id == 1:
    target_name = "mapk14"
else:
    raise ValueError(f"Invalid target_id: {target_id}. Must be 0 or 1.")

config_file = "configs/config.yaml"
with open(config_file) as f:
    cfg = yaml.safe_load(f)

model_type = cfg["model_type"]

# ----------------------------
# Load data
# ----------------------------
df = pd.read_parquet(f"data/processed_{split_type}_s{split}.parquet")
train_df = df[(df["split"] == "train") & (df["target_id"] == target_id)]
val_df = df[(df["split"] == "valid") & (df["target_id"] == target_id)]

if model_type not in ["rf", "xgboost"]:
    num_epochs = cfg["num_epochs"]

num_cores = os.cpu_count()

output_file = f"{target_name}_{model_type}_s{split}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_type in ["rf", "xgboost"]:
    device = torch.device("cpu")

model_rng_state = torch.load("models/model_rng_state.pt")
torch.set_rng_state(model_rng_state)

model = get_model(cfg.get("model_type", "mlp"), cfg)

# ============================================================
# RF / XGBOOST 
# ============================================================
if model_type in ["rf", "xgboost"]:

    train_loader = get_dataloader(train_df, batch_size=len(train_df), num_workers=num_cores, shuffle=False)
    val_loader = get_dataloader(val_df, batch_size=len(val_df), num_workers=num_cores, shuffle=False)

    X_train, y_train = sklearn_concat(train_loader)
    X_val, y_val = sklearn_concat(val_loader)

    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)

    import joblib
    joblib.dump(model.model, f"models/{split_type}/{output_file}.joblib")


# ============================================================
# MLP (PyTorch training)
# ============================================================
else:

    model = model.to(device)

    train_loader = get_dataloader(train_df, batch_size=cfg["batch_size"], num_workers=8, fp_path="data/fp")
    val_loader = get_dataloader(val_df, batch_size=cfg["batch_size"], num_workers=8, fp_path="data/fp")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": []
    }
    
    try:
        for epoch in range(num_epochs):

            model.train()
            train_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

                y_true = batch["enrichment"].float()
                X = batch["ecfp"].to(device)

                y_pred = model(X)
                #print(y_pred.squeeze(-1), y_true)
                loss = F.mse_loss(y_pred.squeeze(-1), y_true)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # validation
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

                    y_true = batch["enrichment"].float()
                    X = batch["ecfp"].to(device)

                    y_pred = model(X)
                    loss = F.mse_loss(y_pred.squeeze(-1), y_true)

                    val_loss += loss.item()


            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss/len(train_loader))
            history["val_loss"].append(val_loss/len(val_loader))

            print(
                f"[{model_type}] "
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train {train_loss/len(train_loader):.4f} | "
                f"Val {val_loss/len(val_loader):.4f}"
            )

            scheduler.step()

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch+1}. Saving current model...")
    
    # save losses
    os.makedirs(f"logs/{split_type}", exist_ok=True)
    os.makedirs(f"models/{split_type}", exist_ok=True)
    loss_df = pd.DataFrame(history)
    loss_df.to_csv(f"logs/{split_type}/{output_file}_loss.csv", index=False)

    # save torch head
    torch.save(model.state_dict(), f"models/{split_type}/{output_file}.pt")
