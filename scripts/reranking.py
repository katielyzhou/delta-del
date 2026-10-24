import pandas as pd
import csv
import yaml
import torch
import joblib
import numpy as np
import argparse
from scipy.stats import spearmanr
import os
import sys
sys.path.append(".")

from delta_del.dataloader import Collater
from delta_del.helpers import set_seed, compute_ecfp
from delta_del.models import get_model

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from rdkit import DataStructs

DEFAULT_TRAIN_TARGETS = ["flt3", "abl2", "ddr1", "ddr2", "abl1", "mapk11", "mapk14", "aurka", "cdk2"]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split-type", type=str, default="random", choices=["random", "disynthon"])
    parser.add_argument("--target-id", type=int, required=True, choices=[0, 1], help="Target ID: 0 for DDR1, 1 for MAPK14")
    parser.add_argument("--train-targets", type=str, nargs="+", default=DEFAULT_TRAIN_TARGETS, help= "ChEMBL proteins to train on")
    parser.add_argument("--data-type", type=str, choices=["ondna", "offdna"], default="offdna",
                        help="Evaluation data type: ondna or offdna")
    parser.add_argument("--in-lib", action="store_true", help="If flag supplied, then consider in-library evaluation only.")
    parser.add_argument("--threshold", type=float, default=1.0, help="Exclude training molecules above the nearest-evaluation threshold.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default=None)
    
    return parser.parse_args()

args = parse_args()
set_seed(123)

split_type = args.split_type
target_id = args.target_id
eval_data_type = args.data_type
in_lib = args.in_lib
data_dir = args.data_dir
train_targets = args.train_targets

threshold = args.threshold # tanimoto similarity threshold for excluding training molecules
data_type = "chembl_Kd" # suffix for training data

config_file = "configs/config.yaml"
with open(config_file) as f:
    cfg = yaml.safe_load(f)
model_type = cfg["model_type"]

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
        }

        if "enrichment" in df.columns:
            sample["enrichment"] = torch.tensor(row.enrichment, dtype=torch.float32)

        samples.append(sample)

    return Collater()(samples)

def get_predictions(model, device, model_type, loader=None, df=None, return_features=False):
    all_preds = []
    all_targets = []
    all_embeddings = []

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
            all_embeddings.append(batch["ecfp"])

    if return_features:
        return np.concatenate(all_targets), np.concatenate(all_preds), np.concatenate(all_embeddings)

    return np.concatenate(all_targets), np.concatenate(all_preds)


device = torch.device("cpu")


final_results = []

target_name = target_map.get(target_id, target_id)
print(f"\n--- Target: {target_name} ---")

for train_target in train_targets:

    train_file = f"{data_dir}/{train_target}_{data_type}.csv"
    
    if not os.path.exists(train_file):
        print(f"Skipping {train_target}: file not found ({train_file})")
        continue
    
    for delta in [True, False]: # with delta learning or direct binding affinity prediction

        for w_poisson in [True, False]: # y_DEL as input feature, which in this case is the Poisson enrichment

            ranking_model = {}

            results = {"base": [], "re": []}

            if not delta and not w_poisson:
                split_indices = [1] # no DEL model is used here
            else:
                split_indices = [1, 2, 3, 4, 5]

            for split_idx in split_indices:

                num_cores = os.cpu_count()

                df = pd.read_csv(train_file)

                train_df = adapt_csv(df, target_id=target_id)

                # Load the evaluation dataset
                target_name = target_map[target_id]
                test_df = pd.read_csv(f"data/{target_name.lower()}_{eval_data_type}.csv")

                if in_lib:
                    test_df = test_df.dropna(subset=["molecule_hash"])

                test_df = adapt_csv(test_df, target_id=target_id)

                # Calculate molecular fingerprints
                train_fps = [
                    compute_ecfp(s, return_as="fp")
                    for s in train_df["smiles"]
                ]

                test_fps = [
                    compute_ecfp(s, return_as="fp")
                    for s in test_df["smiles"]
                ]

                # Keep track of valid test fingerprints
                test_fps = [fp for fp in test_fps if fp is not None]

                if not test_fps:
                    raise ValueError("No valid molecular fingerprints in the test set.")

                # Calculate nearest-neighbour similarity for every training molecule
                train_max_sim = []

                for train_fp in train_fps:
                    if train_fp is None:
                        train_max_sim.append(np.nan)
                        continue

                    similarities = DataStructs.BulkTanimotoSimilarity(
                        train_fp,
                        test_fps
                    )

                    train_max_sim.append(max(similarities))

                train_max_sim = np.array(train_max_sim)

                # Filter training molecules by similarity threshold
                # Exclude molecules with similarity >= threshold to any test molecule
                keep = (
                    np.isfinite(train_max_sim)
                    & (train_max_sim < threshold)
                )

                train_df = train_df.loc[keep].reset_index(drop=True)

                print(f"Training molecules after similarity filtering: {len(train_df)}")
                print(f"Similarity threshold: {threshold}")


                # Save training molecule counts for threshold analysis if threshold is specified
                if threshold < 1.0:
                    counts_file = f"results/{split_type}/training_molecule_counts.csv"
                    os.makedirs(os.path.dirname(counts_file), exist_ok=True)

                    count_row = {
                        "threshold": threshold,
                        "train": train_target.upper(),
                        "eval": target_name.upper(),
                        "n_molecules": len(train_df)
                    }

                    if os.path.exists(counts_file):
                        counts_df = pd.read_csv(counts_file)

                        # Remove the existing matching row before updating it
                        counts_df = counts_df[
                            ~(
                                (counts_df["threshold"] == threshold) &
                                (counts_df["train"] == train_target.upper()) &
                                (counts_df["eval"] == target_name.upper())
                            )
                        ]

                        counts_df = pd.concat(
                            [counts_df, pd.DataFrame([count_row])],
                            ignore_index=True
                        )
                    else:
                        counts_df = pd.DataFrame([count_row])

                    counts_df.to_csv(counts_file, index=False)


                model = get_model(cfg.get("model_type", "mlp"), cfg)
                
                model_file = f"{target_name.lower()}_{model_type}_s{split_idx}"

                if "mlp" in model_type:
                    model.load_state_dict(
                        torch.load(f"models/{split_type}/{model_file}.pt", map_location=device)
                    )
                else:
                    model.model = joblib.load(f"models/{split_type}/{model_file}.joblib")
        

                # ----------------------------
                # predictions
                # ----------------------------
                y_true, y_pred_base, X = get_predictions(
                    model, device, model_type, df=train_df, return_features=True
                )

                y_true = -1 * np.log1p(y_true)

                # ----------------------------
                # features
                # ----------------------------
                S = y_pred_base.reshape(-1, 1)

                if not w_poisson:
                    Z = X
                else:
                    Z = np.hstack([S, X])

                # ----------------------------
                # residual target 
                # ----------------------------
                if delta:
                    y_residual = y_true - y_pred_base
                else:
                    y_residual = y_true

                # ----------------------------
                # Gaussian Process model
                # ----------------------------
                
                kernel = (
                    ConstantKernel(1.0, (1e-3, 1e3)) *
                    RBF(length_scale=1.5, length_scale_bounds=(1.5, 20)) +
                    WhiteKernel(noise_level=5.0, noise_level_bounds=(1e-2, 10.0))
                )
                
                gpr = GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    n_restarts_optimizer=5,
                    random_state=42
                )

                gpr.fit(Z, y_residual)
                #print(gpr.kernel_)

                # ----------------------------
                # prediction
                # ----------------------------
                residual_pred, residual_std = gpr.predict(Z, return_std=True)

                if delta:
                    y_score = y_pred_base + residual_pred
                else:
                    y_score = residual_pred

                # ----------------------------
                # evaluation
                # ----------------------------
                rho_base, _ = spearmanr(y_true, y_pred_base)
                rho_final, _ = spearmanr(y_true, y_score)

                results["re"].append(rho_final)
                results["base"].append(rho_base)
                ranking_model[split_idx] = gpr

                        
            print(f"\n--- RERANKED TRAINING RESULTS: {target_name} ---")
            print(f"Trained on {train_target}, delta {delta}, w_poisson: {w_poisson}")
            print("Spearman (base):", np.mean(results["base"]), "±", np.std(results["base"]))
            print("Spearman (reranked):", np.mean(results["re"]),"±", np.std(results["re"]))


            # ----------------------------
            # Evaluation
            # ----------------------------
            results = {"base": [], "re": []}

            for split_idx in split_indices:

                with open(config_file) as f:
                    cfg = yaml.safe_load(f)

                num_cores = os.cpu_count()

                df = pd.read_csv(f"data/{target_name.lower()}_{eval_data_type}.csv")
                if in_lib:
                    df = df.dropna(subset="molecule_hash")
                    
                test_df = adapt_csv(df, target_id = target_id)

                # ----------------------------
                # Load model
                # ----------------------------
                model = get_model(cfg.get("model_type", "mlp"), cfg)
                                
                model_file = f"{target_name.lower()}_{model_type}_s{split_idx}"

                if "mlp" in model_type:
                    model.load_state_dict(
                        torch.load(f"models/{split_type}/{model_file}.pt", map_location=device)
                    )
                else:
                    model.model = joblib.load(f"models/{split_type}/{model_file}.joblib")

                # ----------------------------
                # predictions
                # ----------------------------
                y_true, y_pred_base, X = get_predictions(
                                model, device, model_type, df=test_df, return_features=True
                            )
                y_true = -1 * np.log1p(y_true)

                S = y_pred_base.reshape(-1, 1)

                if not w_poisson:
                    Z = X
                else:
                    Z = np.hstack([S, X])
                
                rank_model = ranking_model[split_idx]
                residual_pred, residual_std = rank_model.predict(Z, return_std=True)

                if delta:
                    y_score = y_pred_base + residual_pred
                else:
                    y_score = residual_pred


                rho_base, _ = spearmanr(y_true, y_pred_base)
                rho_final, _ = spearmanr(y_true, y_score)

                results["re"].append(rho_final)
                results["base"].append(rho_base)
        

                        
            print(f"\n--- RERANKED TEST RESULTS: {target_name} ---")
            print(f"Trained on {train_target}, delta {delta}, w_poisson: {w_poisson}")
            print("Spearman (base):", np.mean(results["base"]), "±", np.std(results["base"]))
            print("Spearman (reranked):", np.mean(results["re"]),"±", np.std(results["re"]))
            
            final_results.append(
                {
                    "threshold":threshold,
                    "train":train_target.upper(),
                    "train_type": data_type,
                    "eval":target_name.upper(),
                    "eval_type":eval_data_type,
                    "delta":delta,
                    "w_poisson":w_poisson,
                    "result":np.mean(results["re"]),
                    "result_std":np.std(results["re"])
                }
            )

results_df = pd.DataFrame(final_results).drop_duplicates().reset_index(drop=True)

threshold_suffix = f"_threshold{threshold:.1f}"

if args.output_dir is None:
    output_directory = f"results/{split_type}"
else:
    output_directory = f"results/{split_type}/" + args.output_dir

os.makedirs(output_directory, exist_ok=True)

output_file = output_directory + f"/reranking_results_{target_name}{threshold_suffix if threshold < 1 else ''}.csv"

results_df.to_csv(output_file, index=False)

print(f"Results saved to: {output_file}")
print(results_df)
