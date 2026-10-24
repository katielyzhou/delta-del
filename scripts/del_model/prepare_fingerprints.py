import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import sys
sys.path.append(".")

from delta_del.helpers import compute_ecfp


# ----------------------------
# Helpers
# ----------------------------
def get_unique_molecules(input_files):
    dfs = []
    for f in input_files:
        df = pd.read_parquet(f, columns=["molecule_hash", "smiles"])
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    return full.drop_duplicates("molecule_hash", keep="first").reset_index(drop=True)


def chunkify(smiles, hashes, chunk_size=200):
    for i in range(0, len(smiles), chunk_size):
        yield (
            smiles[i:i + chunk_size],
            hashes[i:i + chunk_size]
        )


# ----------------------------
# Worker (batch-based)
# ----------------------------
def process_batch(batch):
    smiles, hashes = batch

    n = len(smiles)

    ecfp_all = np.zeros((n, 1024), dtype=np.uint8)

    out_hashes = []

    for i, smi in enumerate(smiles):
        try:

            ecfp_all[i] = compute_ecfp(smi, return_as = "np")

            out_hashes.append(hashes[i])

        except Exception:
            continue

    return ecfp_all, out_hashes


# ----------------------------
# Main
# ----------------------------
def process_files(input_files, output_prefix, n_workers=None):

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"Using {n_workers} workers")

    df = get_unique_molecules(input_files)

    smiles = df["smiles"].values
    hashes = df["molecule_hash"].values

    n = len(df)
    print(f"Unique molecules: {n}. Please update FingerprintDB in delta_del/dataloader.py if not 1875955.")

    # ----------------------------
    # MEMMAP OUTPUT
    # ----------------------------
    fp_path = output_prefix + "_fp.npy"

    fp = np.memmap(
        fp_path,
        dtype=np.uint8,
        mode="w+",
        shape=(n, 1, 1024)
    )

    hash_list = []

    chunks = list(chunkify(smiles, hashes, chunk_size=200))

    idx = 0

    # ----------------------------
    # MULTIPROCESSING
    # ----------------------------
    with Pool(
        processes=n_workers,
        maxtasksperchild=50
    ) as pool:

        for ecfp, hbatch in tqdm(
            pool.imap(process_batch, chunks),
            total=len(chunks)
        ):

            bsz = len(hbatch)

            if bsz == 0:
                continue

            fp[idx:idx + bsz, 0] = ecfp[:bsz]

            hash_list.extend(hbatch)
            idx += bsz

    fp.flush()

    hash_list = np.array(hash_list, dtype=str)

    hash_to_idx = {h: i for i, h in enumerate(hash_list)}

    np.save(output_prefix + "_hashes.npy", hash_list)

    import pickle
    with open(output_prefix + "_hash_to_idx.pkl", "wb") as f:
        pickle.dump(hash_to_idx, f)

    print(f"Saved fingerprint DB → {fp_path}")


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", default="data/fp")
    parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()

    process_files(args.input, args.output, args.workers)


if __name__ == "__main__":
    main()
