import pandas as pd
import csv
import yaml
import argparse
import torch
torch.set_printoptions(threshold=float("inf"))
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import os
import sys
sys.path.append(".")

from delta_del.dataloader import get_dataloader, Collater
from delta_del.helpers import set_seed, compute_ecfp
from delta_del.models import get_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split-type", type=str, default="random", choices=["random", "disynthon"])
    parser.add_argument("--target-id", type=int, required=True, choices=[0, 1], help="Target ID: 0 for DDR1, 1 for MAPK14")
    parser.add_argument("--data-type", type=str, required=True, choices=["ondna", "offdna"], help="Data type: ondna or offdna")
    parser.add_argument("--in-lib", action="store_true", help="If flag supplied, then consider in-library evaluation only.")


    return parser.parse_args()

# ----------------------------
# Setup
# ----------------------------
set_seed(123)
model_rng_state = torch.load("models/model_rng_state.pt")
torch.set_rng_state(model_rng_state)

args = parse_args()

split_type = args.split_type
target_id = args.target_id
data_type = args.data_type
in_lib = args.in_lib

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

splits = [1, 2, 3, 4, 5]

base_name = f"{model_type}_{data_type}_regress_metrics_log.csv"

target_map = {0: "DDR1", 1: "MAPK14"}


def adapt_csv(df, target_id=0):

    # --- REQUIRED ---
    df = df.copy()

    df["target_id"] = target_id

    if "kd" in df.columns:
        df["enrichment"] = df["kd"]

    return df

def build_batch(df):
    """
    Convert DEL-style dataframe into a list of samples compatible with `collater`.
    """

    samples = []

    for row in df.itertuples(index=False):
        sample = {
            "ecfp": compute_ecfp(row.smiles, nBits=1024),
            "target_id": torch.tensor(row.target_id, dtype=torch.long),
            #"replicate_id": torch.tensor(row.replicate_id, dtype=torch.long),
        }

        if "enrichment" in df.columns:
            sample["enrichment"] = torch.tensor(row.enrichment, dtype=torch.float32)

        samples.append(sample)

    return Collater()(samples)


def get_predictions(model, device, model_type, loader=None, df=None):
    all_preds = []
    all_targets = []

    model.eval() if "mlp" in model_type else None

    with torch.no_grad():
        if loader is not None:
            for batch in loader:
                batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                y_true = batch["enrichment"].float().cpu().numpy()
                X = batch["ecfp"].to(device)                  # [B, D]
                out = model(X)

                # ----------------------------
                # unify output
                # ----------------------------
                if "mlp" in model_type:
                    preds = out.cpu().numpy().squeeze()
                else:
                    preds = np.asarray(out)

                all_preds.append(preds)
                all_targets.append(y_true)

        else:
            batch = build_batch(df)
            batch = {
                        k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }

            y_true = batch["enrichment"].float().cpu().numpy()
            X = batch["ecfp"].to(device)                  # [B, D]
            out = model(X)

            # ----------------------------
            # unify output to probabilities
            # ----------------------------
            if "mlp" in model_type:
                preds = out.cpu().numpy().squeeze()
            else:
                preds = np.asarray(out)

            all_preds.append(preds)
            all_targets.append(y_true)

    return np.concatenate(all_targets), np.concatenate(all_preds)


# ----------------------------
# Main loop
# ----------------------------
device = torch.device("cpu")
rhos = []

for idx in splits:

    print(f"\n===== SPLIT {idx} =====")

    num_cores = os.cpu_count()

    target_name = target_map.get(target_id, target_id)

    print(f"\n--- Target: {target_name} ---")

    if data_type == "test":
        df = pd.read_parquet(f"data/processed_{split_type}_s{idx}.parquet")

        test_df = df[(df["split"] == "test") & (df["target_id"] == target_id)]
        test_loader = get_dataloader(
            test_df,
            batch_size=cfg["batch_size"]
            if "mlp" in model_type
            else len(test_df),
            num_workers=num_cores,
        )

    else:
        df = pd.read_csv(f"data/{target_name.lower()}_{data_type}.csv")
        if in_lib:
            df = df.dropna(subset="molecule_hash")

        test_df = adapt_csv(df, target_id = target_id)
        test_loader = None

    model = get_model(cfg.get("model_type", "mlp"), cfg)

    model_file = f"{target_name.lower()}_{model_type}_s{idx}"
        

    if "mlp" in model_type:
        model.load_state_dict(
            torch.load(f"models/{split_type}/{model_file}.pt", map_location=device)
        )
    else:
        model.model = joblib.load(f"models/{split_type}/{model_file}.joblib")


    # ----------------------------
    # predictions
    # ----------------------------
    if test_loader is not None:
        y_true, y_pred = get_predictions(
            model, device, model_type, loader=test_loader
        )
    else:
        y_true, y_pred = get_predictions(
            model, device, model_type, df=test_df
        )

    # ----------------------------
    # metrics
    # ----------------------------

    rho, pval = spearmanr(y_true, y_pred)
    rhos.append(rho)

    print(rho)

print(f"\n--- FINAL RESULTS: {target_name} ---")
print("Spearman:", np.mean(rhos), "±", np.std(rhos))

