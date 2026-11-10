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

# Collate

def collate(batch):
    specs, tokens = zip(*batch) # seperate the image from the token list
    n_bins = specs[0].shape[0] 
    max_frames = max(s.shape[1] for s in specs) # longest possible clip in the batch
    max_tokens = max(len(t) for t in tokens) # longest possible token list in the batch

    spec_batch = torch.zeros(len(batch), n_bins, max_frames) # empty normalization
    frame_mask = torch.zeros(len(batch), max_frames, dtype=torch.bool) # ignores filler
    token_batch = torch.full((len(batch), max_tokens), tokenizer.PAD, dtype=torch.long) # adds filler PAD

    for i, (spec, toks) in enumerate(zip(specs, tokens)): # copies each clip into its correct row
        spec_batch[i, :, : spec.shape[1]] = spec
        frame_mask[i, : spec.shape[1]] = True
        token_batch[i, : len(toks)] = toks

    return {
        "spec": spec_batch,
        "frame_mask": frame_mask,
        "decoder_input": token_batch[:, :-1], # tokens without the last one
        "target": token_batch[:, 1:], # tokens without the first one
    }

# Loader

def make_loader(prepared_dir: str, split: str, batch_size: int, workers: int = 4) -> DataLoader:
    is_train = split == "train" # training gets special
    dataset = ElectroDataset(prepared_dir, split, augment=is_train) # random tone for training
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, # mixup the order of epoch
                      num_workers=workers, collate_fn=collate, drop_last=is_train, # load multiple batches to the CPU while GPU does the current one
                      persistent_workers=workers > 0) # keep workers alive inbetween epochs

# Main

if __name__ == "__main__":
    import sys
    prepared = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    loader = make_loader(prepared, "train", batch_size=batch_size, workers=4) # loads training data
    print(f"train clips: {len(loader.dataset)}  batches per epoch: {len(loader)}")

    batch = next(iter(loader)) # chooses a singular batch
    for name, value in batch.items(): # prints the shape of everything in the batch
        print(f"{name:14s} {tuple(value.shape)}  {value.dtype}")

    print("decoder input:", [tokenizer.VOCAB[i] for i in batch["decoder_input"][0][:6]]) # prints the frist 6 tokens of each batch
    print("target       :", [tokenizer.VOCAB[i] for i in batch["target"][0][:6]]) # prints the frist 6 tokens of each batch

    started, n = time.time(), 0
    for n, _ in enumerate(loader, start=1): # loads 50 batches and they're timed
        if n == 50:
            break
    seconds = time.time() - started
    print(f"loading speed: {n * batch_size / seconds:.0f} clips/s") # prints clips per second