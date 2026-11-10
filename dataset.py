"""Electro Dataset: break up the dataset into batches for the model."""

import csv # read the dataset
import json # read the dataset
import time # used to time the run
from pathlib import Path # a simplier way of handling paths

import numpy as np # optimize spectrogram math
import torch # a deep-learning library, runs math on your GPU
from torch.utils.data import DataLoader, Dataset # dataset is used to fetch one example at a time, and dataloader turns dataset into suffled batches

import tokenizer # imports tokenizer.py

BINS_PER_OCTAVE = 36 # tilts the tone

# Fetcher

class ElectroDataset(Dataset): # fetches clip and token
    def __init__(self, prepared_dir: str, split: str, augment: bool = False): # categorizes the clips into train, val, or test
        self.dir = Path(prepared_dir)
        with open(self.dir / "manifest.csv", newline="") as f: # opens manifest.csv
            self.items = [row for row in csv.DictReader(f) if row["split"] == split] # reads the manifest and only keeps what's needed
        stats = json.loads((self.dir / "stats.json").read_text())
        self.mean = np.array(stats["mean"], dtype=np.float32)[:, None] # load the normalization stats into rows
        self.std = np.array(stats["std"], dtype=np.float32)[:, None] # load the normalization stats into rows
        self.augment = augment # adds variability to training clips

    def __len__(self) -> int: # calculates total examples for torch
        return len(self.items)

    def __getitem__(self, index: int): # fetches example using index
        item = self.items[index]
        spec = np.load(self.dir / item["spec"]).astype(np.float32) # load the picture and switch to full precision for math
        if self.augment: # for training only change the tone a little
            spec = random_tone(spec)
        spec = (spec - self.mean) / self.std # normalize the output
        tokens = np.load(self.dir / item["tokens"]).astype(np.int64) # load the tokens
        return torch.from_numpy(spec), torch.from_numpy(tokens) # hand both back as PyTorch tensors

# Cheap Variation

def random_tone(spec: np.ndarray) -> np.ndarray: # adds traning variation
    gain_db = torch.empty(1).uniform_(-6.0, 6.0).item() # changes decible amount by 6 in any direction
    tilt_db = torch.empty(1).uniform_(-3.0, 3.0).item() # changes brightness of spectrogram by 3dBs in either direction
    octave = np.arange(spec.shape[0], dtype=np.float32) / BINS_PER_OCTAVE - 3.5 # octave
    return spec + gain_db + (tilt_db * octave)[:, None] # apply both variations

