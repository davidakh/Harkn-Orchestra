"""Electro Dataset: break up the dataset into batches for the model."""

import csv # read the dataset
import json # read the dataset
import time # used to time the run
import math # rounding up to fixed sizes
import random # shuffling the batches
from pathlib import Path # a simplier way of handling paths

import numpy as np # optimize spectrogram math
import torch # a deep-learning library, runs math on your GPU
from torch.utils.data import DataLoader, Dataset # dataset is used to fetch one example at a time, and dataloader turns dataset into suffled batches

import tokenizer # imports tokenizer.py

BINS_PER_OCTAVE = 36 # tilts the tone
TOKEN_BUCKETS = [128, 256, 512, 1024]
FRAME_STEP = 64

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
    max_frames = math.ceil(max(s.shape[1] for s in specs) / FRAME_STEP) * FRAME_STEP # longest possible clip in the batch
    max_tokens = bucket_for(max(len(t) for t in tokens) - 1) + 1 # longest possible token list in the batch

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

# Group Lengths

def bucket_for(length: int) -> int:
    return next(size for size in TOKEN_BUCKETS if length <= size)

class BucketBatchSampler:
    def __init__(self, dataset, batch_tokens: int, seed=None):
        self.seed = seed
        self.buckets = {size: [] for size in TOKEN_BUCKETS}
        for index, item in enumerate(dataset.items):
            self.buckets[bucket_for(int(item["n_tokens"]) - 1)].append(index)
        self.batch_sizes = {size: max(1, batch_tokens // size) for size in TOKEN_BUCKETS}

    def __iter__(self):
        rng = random.Random(self.seed)
        batches = []
        for size, indices in self.buckets.items():
            indices = indices[:]
            rng.shuffle(indices)
            n = self.batch_sizes[size]
            batches += [indices[i : i + n] for i in range(0, len(indices), n)]
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return sum(math.ceil(len(ix) / self.batch_sizes[size]) for size, ix in self.buckets.items())

# Loader

def make_loader(prepared_dir: str, split: str, batch_tokens: int, workers: int = 2, limit: int = 0) -> DataLoader:
    is_train = split == "train" # training gets special
    dataset = ElectroDataset(prepared_dir, split, augment=is_train and not limit) # random tone for training
    if limit:
        dataset.items = dataset.items[:limit]
    sampler = BucketBatchSampler(dataset, batch_tokens, seed=None if is_train else 0)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=workers, collate_fn=collate,
                      persistent_workers=workers > 0)

# Main

# Main

if __name__ == "__main__":
    import sys
    prepared = sys.argv[1]
    batch_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    loader = make_loader(prepared, "train", batch_tokens=batch_tokens, workers=2)
    print(f"train clips: {len(loader.dataset)}  batches per epoch: {len(loader)}")
    print("clips per length group:", {size: len(ix) for size, ix in loader.batch_sampler.buckets.items()})

    shapes = set()
    started, clips = time.time(), 0
    for n, batch in enumerate(loader, start=1):
        shapes.add((tuple(batch["spec"].shape), tuple(batch["decoder_input"].shape)))
        clips += batch["spec"].shape[0]
        if n == 200:
            break
    seconds = time.time() - started
    print(f"different batch shapes in {n} batches: {len(shapes)}")
    for spec_shape, token_shape in sorted(shapes)[:6]:
        print(f"  spec {spec_shape}  decoder_input {token_shape}")
    print(f"loading speed: {clips / seconds:.0f} clips/s")