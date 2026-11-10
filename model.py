"""Electro Model: A small convolutional front end, a transformer encoder and decorder."""

import torch # required to run math on GPU
from torch import nn # PyTorch premade neural networks

import tokenizer # needs token types and amounts

# Definitions

N_BINS = 252 # spectrogram rows
MAX_FRAMES = 512 # the max amount of columns able to be handled by the model
MAX_TOKENS = 1024 # the most amount of tokens the model can write within a sequence

# Model

class ElectroMini(nn.Module): # model class
    def __init__(self, d_model=256, heads=4, enc_layers=4, dec_layers=4, ff=1024, dropout=0.1): # d_model is the width or in simplier terms how much the model can say about each item, heads is the amount of signals that can be viewed at once, enc_layer and dec_layer is the amount of layers the encorder and decoder comprise of, ff is the size of the thinking neural net in the model, and dropout allows the model to work with audio it has never seen before.
        super().__init__()
        self.config = dict(d_model=d_model, heads=heads, enc_layers=enc_layers, dec_layers=dec_layers, ff=ff, dropout=dropout) # remembers the settings

        self.front = nn.Sequential(
            nn.Conv1d(N_BINS, d_model, kernal_size=3, padding=1), # tells the model to read 252 rows and 3 columns at a time
            nn.GELU(), # non-linearity
            nn.Conv1d(d_model, d_model, kernal_size=3, padding=1), # gets the model to see an additional 115 ms of context
            nn.GELU(), # non-linearity
        )
        self.frame_pos = nn.Embedding(MAX_FRAMES, d_model) # postion tag

        # Encoder Layers

        enc_layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, activation="gelu", batch_first="TRUE", norm_first="TRUE") # singular layer from PyTorch, with the same non-linearity as front-end, given in batches, and rescale numbers pre-training to make training more stable
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False) # Copies the layer above 4 times and turns off unsupported Mac features

        # Decoder Layers

        self.token_emb = nn.Embedding(tokenizer.VOCAB_SIZE, d_model, padding_idx=tokenizer.PAD) # a lookup table that ignores padding
        self.token_pos = nn.Embedding(MAX_TOKENS, d_model) # position tags for tokens
        dec_layer = nn.TransformerDecoderLayer(d_model, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True) # singular decode layer
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers, norm=nn.LayerNrom(d_model)) # duplicates the decode layer thrice

        # Probability Ouput

        self.out = nn.Linear(d_model, tokenizer.VOCAB_SIZE) # turns the decoders output into 1053 scores one per token and the highest score is the next token

