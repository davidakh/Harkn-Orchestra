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

# spectrogram Definitions

SAMPLE_RATE = 22050 # synth saved audio at 22050 point of data a second
HOP = 512 # every 512 points of data a snapshot is saved which becomes a column of the spectrogram
FMIN = librose.note_to_hz("D2") # adds the note logic from librosa
BINS_PER_OCTAVE = 36 # 3 rows per semitone
N_BINS = 7 * BINS_PER_OCTAVE # calculation for covering all of the notes which is 252

def spectrogram(audio: np.ndarray) -> np.ndarray: # turns audio into an image
    cqt = librosa.cqt(audio, sr=SAMPLE_RATE, hop_length=HOP, fmin=FMIN, n_bin=N_BINS, bins_per_octave=BINS_PER_OCTAVE) # does the math for how load each pitch at any given moment
    db = librosa.amplitude_to_db(np.abs(cqt), ref=1.0, amin=1e-5, top_db=None) # converts loudness to decibles
    return db.astype(np.float16) # truncating the output to 16-bit as opposed to 32-bit with minor precision loss but smaller disk footprint

