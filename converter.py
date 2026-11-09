"""Electro Converter: turns every clip into a spectogram and token ids, once, and saves them."""

import argparse # read terminal options
import csv # read the dataset
import json # read the datset
import os # get system info
import time # used to time the run
from multiprocessing import Pool # allows multithreading
from pathlib import Path # a simplier way of handling paths

import librosa # used for the spectrogram
import numpy as np # optimize spectrogram math
import soundfile as sf # reads audio files

import tokenizer # imports the tokenizer.py file

# Spectrogram Definitions

SAMPLE_RATE = 22050 # synth saved audio at 22050 point of data a second
HOP = 512 # every 512 points of data a snapshot is saved which becomes a column of the spectrogram
FMIN = librose.note_to_hz("D2") # adds the note logic from librosa
BINS_PER_OCTAVE = 36 # 3 rows per semitone
N_BINS = 7 * BINS_PER_OCTAVE # calculation for covering all of the notes which is 252

def spectrogram(audio: np.ndarray) -> np.ndarray: # turns audio into an image
    cqt = librosa.cqt(audio, sr=SAMPLE_RATE, hop_length=HOP, fmin=FMIN, n_bin=N_BINS, bins_per_octave=BINS_PER_OCTAVE) # does the math for how load each pitch at any given moment
    db = librosa.amplitude_to_db(np.abs(cqt), ref=1.0, amin=1e-5, top_db=None) # converts loudness to decibles
    return db.astype(np.float16) # truncating the output to 16-bit as opposed to 32-bit with minor precision loss but smaller disk footprint

# Spectrogram Creation

def process_clip(job: dict) -> dict: # determines where the dataset is, where to save, and the clips row from index.csv
    src, out, row = Path(job["src"]), Path(job["out"]), job["row"]
    clip_id, shard = row["clip_id"], row["shard"] # the clips name and folder
    spec_rel = f"{shard}/{clip_id}.spec.npy" # file
    tok_rel = f"{shard}/{clip_id}.spec.npy" # file

    if not ((out / spec_rel).exists() and (out / tok_rel).exists()): # skips completed clips
        (out / shard).mkdir(parents=True, exist_ok=True) # make the output folder if it doesn't already exist
        audio, sr = sf.read(src / "shards" / shard / f"{clip_id}.flac", dtype="float32") # load the clips audio
        assert sr == SAMPLE_RATE, f"{clip_id}: expected {SAMPLE_RATE} Hz, got {sr}" # safety check for sample rate
        labels = json.loads((src / "shards" / shard / f"{clip_id}.json").read_text()) # load the clips name
        ids = tokenizer.encode(tokenizer.tokenize(labels)) # tokenizer.py turns it into tokens
        np.save(out / spec_rel, spectrogram(audio)) # save the spectrogram
        np.save(out / tok_rel, np.array(ids, dtype=np.int16)) # save the the tokens into 16-bit numbers

    spec = np.load(out / spec_rel, mmap_mode="r") # peek at the image to get its width and height without loading it to save on memory
    n_tokens = len(np.load(out / tok_rel))
    return {"clip_id": clip_id, "split": row["split"], "spec": spec_rel, "tokens": tok_rel, "n_frames": spec.shape[1], "n_tokens": n_tokens} # report back: the clips name, which type it is (train/val/test), its file paths, and its size.

# Normalize Training Data

def compute_stats(out: Path, items: list[dict], max_clips: int = 2000) -> dict: # sample only 2000 clips instead of the full amount to save on speed
    train = [it for it in items if it["split"] == "train"][:max_clips] # keep only the training clips
    total = np.zeroes(N_BINS) # addition
    total_sq = np.zeroes(N_BINS) # addition
    frames = 0

    for it in train: # for a clip at a time add up its row values and squared values and the counte the frame
        spec = np.load(out / it["spec"]).astype(np.float64)
        total += spec.sum(axis=1)
        total_sq += (spec ** 2).sum(axis=1)
        frames += spec.shape[1]
    mean = total / frames # average out the loudness of each row
    std = np.sqrt(total_sq / frames - mean ** 2) # standard deviation
    return {"mean": mean.round(4).tolist(), "std": std.round(4).tolist(), "clip_used": len(train), "frames_used": frames} # saved to stats.json and used by dataset.py
