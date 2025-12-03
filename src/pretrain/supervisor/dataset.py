import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

LABELS = ["none", "low", "medium", "high"]
LABEL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LABEL = {i: k for k, i in LABEL2ID.items()}
LABEL_SET = set(LABELS)

class CTQEDataset(Dataset):
    """
    Reads QE dataset_info.csv with columns:
      image_paths, ldct_severities, lact_severities, svct_severities, degradations
    """

    def __init__(
        self,
        dataset_info_file: str,
        split: str,
        train_ratio: float,
        seed: int,
    ):
        assert split in ("train", "test")
        self.df = pd.read_csv(dataset_info_file).reset_index(drop=True)

        n = len(self.df)
        idx = np.arange(n)
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        n_train = int(round(n * train_ratio))
        sel = idx[:n_train] if split == "train" else idx[n_train:]

        self.df = self.df.iloc[sel].reset_index(drop=True)
        
        # Helper function for cleaning severity strings
        def _clean_sev_local(x):
            if x is None:
                return "none"
            if isinstance(x, float) and np.isnan(x):
                return "none"
            s = str(x).strip().lower()
            if s in ("", "nan", "null", "none"):
                return "none"
            return s

        for col in ["ldct_severities", "lact_severities", "svct_severities"]:
            self.df[col] = self.df[col].apply(_clean_sev_local)
            bad = ~self.df[col].isin(LABEL_SET)
            if bad.any():
                raise ValueError(f"Invalid values in {col}: {self.df.loc[bad, col].head(10).tolist()}")


    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = np.load(row["image_paths"])

        x = torch.from_numpy(img).squeeze(0)  # [1,H,W]

        y = {
            "ldct": torch.tensor(LABEL2ID[row["ldct_severities"]], dtype=torch.long),
            "lact": torch.tensor(LABEL2ID[row["lact_severities"]], dtype=torch.long),
            "svct": torch.tensor(LABEL2ID[row["svct_severities"]], dtype=torch.long),
        }
        return x, y
