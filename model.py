"""Electro Model: A small convolutional front end, a transformer encoder and decorder."""

import torch # required to run math on GPU
from torch import nn # PyTorch premade neural networks

import tokenizer # needs token types and amounts

# Definitions

N_BINS = 252 # spectrogram rows
MAX_FRAMES = 512 # the max amount of columns able to be handled by the model
MAX_TOKENS = 1024 # the most amount of tokens the model can write within a sequence

#