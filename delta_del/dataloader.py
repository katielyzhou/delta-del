import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle

class Collater:
    def __call__(self, batch):
        collated = {}
        for key in batch[0]:
            collated[key] = torch.stack([b[key] for b in batch])
        return collated


class FingerprintDB: # Pre-computed FP
    def __init__(self, prefix, n_fp=1):
        self.fp = np.memmap(
            prefix + "_fp.npy",
            mode="r",
            shape=(1875955, n_fp, 1024) # number of unique molecules from prepare_fingerprints.py, number of fp types (e.g. ecfp), fp bits
        )
        self.hashes = np.load(prefix + "_hashes.npy")

        with open(prefix + "_hash_to_idx.pkl", "rb") as f:
            self.hash_to_idx = pickle.load(f)

    def get(self, h):
        i = self.hash_to_idx[h]
        return self.fp[i]   # (1, 1024)
    

class DELDataset(Dataset):
    def __init__(self, df: pd.DataFrame, fp_db: FingerprintDB, nbits: int = 1024):

        self.df = df.reset_index(drop=True)
        self.nbits = nbits
        self.fp_db = fp_db

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        fp = self.fp_db.get(row["molecule_hash"])
        ecfp = torch.from_numpy(fp[0].copy()).float()

        sample = {
            'ecfp': ecfp,
            'target_id': torch.tensor(row['target_id'], dtype=torch.long),
            #'replicate_id': torch.tensor(row['replicate_id'], dtype=torch.long),
            'enrichment': torch.tensor(row['enrichment'], dtype=torch.float32),
        }

        return sample

def get_dataloader(df: pd.DataFrame, fp_path="data/fp", batch_size=128, shuffle=True, num_workers=0, nbits=1024):
    fp_db = FingerprintDB(fp_path)
    dataset = DELDataset(df, fp_db, nbits=nbits)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=Collater(),
        pin_memory=True
    )
