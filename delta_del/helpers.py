import random
import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit import RDLogger
# Suppress RDKit warnings (including deprecation warnings)
RDLogger.DisableLog('rdApp.*')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def compute_ecfp(smiles: str, radius=2, nBits=1024, return_as = "tensor", useChirality=True) -> torch.Tensor:
    """
    Compute Morgan (ECFP) fingerprint from SMILES.
    return_as = tensor, np, fp
    """
    mol = Chem.MolFromSmiles(smiles)
    arr = np.zeros(nBits, dtype=np.float32)
    
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits, useChirality=useChirality)

        if return_as == "fp":
            return fp

        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)

        if return_as == "tensor":
            return torch.tensor(arr, dtype=torch.float32)
        elif return_as == "np":
            return arr.astype(np.uint8)
        else:
            raise ValueError(f"Invalid return type. tensor, np (array), or (rdkit) fp")
    else:
        return None

    
