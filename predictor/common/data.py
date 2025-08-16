# common/data.py
import os
import random
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NormalDataset(Dataset):
    def __init__(self, data: List[List[int]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        input_seq = torch.tensor(row[:7], dtype=torch.long)
        label = torch.tensor(row[7], dtype=torch.long)
        return input_seq, label

def load_data(data_path: str, n_rows: int, batch_size: int):
    """Loads data from CSV, splits it, and returns DataLoaders."""
    print(f"Loading up to {n_rows} rows from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: Data file not found at '{data_path}'")

    data = list(np.loadtxt(data_path, dtype=np.int64, delimiter=',', max_rows=n_rows))
    random.shuffle(data)

    train_end = int(0.8 * len(data))
    val_end = int(0.9 * len(data))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print(f"Dataset split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples.")

    train_loader = DataLoader(NormalDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NormalDataset(val_data), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(NormalDataset(test_data), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader