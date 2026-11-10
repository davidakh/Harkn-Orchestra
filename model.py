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

    # Encode

    def encode(self, spec, frame_mask): # encodes the tokens
        x = self.front(spec) # runs the convolutions
        x = x.transpose(1, 2) # swap batches and give some time in between
        positions = torch.arange(x.shape[1], device=x.device) # increase the number of columns
        x = x + self.frame_pos(positions) # columns position tags
        return self.encoder(x, src_key_padding_mask=~frame_mask) # ignores padding

    # Decode

    def decode(self, tokens, memory, frame_mask): # decodes the tokens
        n = tokens.shape[1]
        positions = torch.arange(n, device=tokens.device) # lookup a token's meaning and position
        x = self.token_emb(tokens) + self.token_pos(positions)
        future = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tokens.device), diagonal=1) # prevents the model from seeing the next token it's trying to predict
        x = self.decoder(x, memory, tgt_mask=future, tgt_key_padding_mask=tokens == tokenizer.PAD, memory_key_padding_mask=~frame_mask) # ignores padding in tokens and audio
        return self.out(x) # returns prediction for the tokens

    def forward(self, spec, frame_mask, decoder_input): # run loop for training
        memory = self.encode(spec, frame_mask) # listens to a singular clip in its full
        return self.decode(decoder_input, memory, frame_mask) # write all predictions at once

# Generate Loop

@torch.no_grad() # disable learning
def generate(self, spec, frame_mask, max_tokens=MAX_TOKENS): # generates tab
    memory = self.encode(spec, frame_mask) # listens once to the clip
    batch = spec.shape[0]
    tokens = torch.full((batch, 1), tokenizer.SOS, dtype=torch.long, device=spec.device) # wraps output
    done = torch.zeros(batch, dtype=torch.bool, device=spec.device) # tracks which clips are finished
    for _ in range(max_tokens - 1): # loop for writing until limit
        scores = self.decode(tokens, memory, frame_mask) # scores every possible next tokens
        next_token = scores[:, -1].argmax(dim=-1) # greedy decoding
        next_token[done] = tokenizer.PAD # clips that are finished get padding
        tokens = torch.cat([tokens, next_token[:, None]], dim=1) # adds the predicted token to the end
        done |= next_token == tokenizer.EOS # once output is finished wrap in end of sequence token
        if done.all(): # stop early if every clip is finished
            break
    return tokens # return predicted tokens

# Test

if __name__ == "__main__":
    model = ElectroMini()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.1f} million")

    spec = torch.randn(2, N_BINS, 431)
    frame_mask = torch.ones(2, 431, dtype=torch.bool)
    decoder_input = torch.randint(3, tokenizer.VOCAB_SIZE, (2, 50))
    scores = model(spec, frame_mask, decoder_input)
    print(f"input  spec {tuple(spec.shape)}  tokens {tuple(decoder_input.shape)}")
    print(f"output scores {tuple(scores.shape)}  (one score per vocabulary token, at every position)")

    model.eval()
    written = model.generate(spec[:1], frame_mask[:1], max_tokens=10)
    print("untrained model writes:", [tokenizer.VOCAB[i] for i in written[0].tolist()])