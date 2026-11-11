"""Electro Model: A small convolutional front end, a transformer encoder and decorder."""

import mlx.core as mx # required to run math on Apple GPU
import mlx.nn as nn # MLX premade neural networks
from mlx.utils import tree_flatten # nests weights within organized tree

import tokenizer # needs token types and amounts

# Definitions

N_BINS = 252 # spectrogram rows
MAX_FRAMES = 512 # the max amount of columns able to be handled by the model
MAX_TOKENS = 1024 # the most amount of tokens the model can write within a sequence

# Model

class Electro(nn.Module): # model class
    def __init__(self, d_model=256, heads=4, enc_layers=4, dec_layers=4, ff=1024, dropout=0.1): # d_model is the width or in simplier terms how much the model can say about each item, heads is the amount of signals that can be viewed at once, enc_layer and dec_layer is the amount of layers the encorder and decoder comprise of, ff is the size of the thinking neural net in the model, and dropout allows the model to work with audio it has never seen before.
        super().__init__()

        self.front = nn.Sequential(
            nn.Conv1d(N_BINS, d_model, kernel_size=3, padding=1), # tells the model to read 252 rows and 3 columns at a time
            nn.GELU(), # non-linearity
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), # gets the model to see an additional 115 ms of context
            nn.GELU(), # non-linearity
        )
        self.frame_pos = nn.Embedding(MAX_FRAMES, d_model) # postion tag

    # Encoder Layers

        self.encoder = nn.TransformerEncoder(enc_layers, d_model, heads, ff, dropout, activation=nn.gelu, norm_first=True, checkpoint=checkpoint) # Copies the layer above 4 times and turns off unsupported Mac features

    # Decoder Layers

        self.token_emb = nn.Embedding(tokenizer.VOCAB_SIZE, d_model) # a lookup table that ignores padding
        self.token_pos = nn.Embedding(MAX_TOKENS, d_model) # position tags for tokens
        self.decoder = nn.TransformerDecoder(dec_layers, d_model, heads, ff, dropout, activation=nn.gelu, norm_first=True, checkpoint=checkpoint) # duplicates the decode layer thrice

    # Probability Ouput

        self.out = nn.Linear(d_model, tokenizer.VOCAB_SIZE) # turns the decoders output into 1053 scores one per token and the highest score is the next token

    # Encode

    def encode(self, spec, frame_mask): # encodes the tokens
        x = self.front(spec.transpose(0, 2, 1)) # runs the convolutions
        x = x + self.frame_pos(mx.arange(x.shape[1])) # columns position tags
        return self.encoder(x, frame_mask[:, None, None, :]) # masks the clip

    # Decode

    def decode(self, tokens, memory, frame_mask): # decodes the tokens
        n = tokens.shape[1]
        x = self.token_emb(tokens) + self.token_pos(mx.arange(n))
        past = mx.tril(mx.ones((n, n), dtype=mx.bool_)) # hides parts of the clip
        self_mask = past[None, None] & (tokens != tokenizer.PAD)[:, None, None, :] # allows a token to loook another one as long as its a previously occuring one and isn't padding
        x = self.decoder(x, memory, self_mask, frame_mask[:, None, None, :]) # ignores padding in tokens and audio
        return self.out(x) # returns prediction for the tokens

    def __call__(self, spec, frame_mask, decoder_input):
        memory = self.encode(spec, frame_mask)
        return self.decode(decoder_input, memory, frame_mask)

# Generate Loop

    def generate(self, spec, frame_mask, max_tokens=MAX_TOKENS):
        memory = self.encode(spec, frame_mask)
        batch = spec.shape[0]
        tokens = mx.full((batch, 1), tokenizer.SOS, dtype=mx.int32)
        done = mx.zeros((batch,), dtype=mx.bool_)
        for _ in range(max_tokens - 1):
            scores = self.decode(tokens, memory, frame_mask)
            next_token = mx.argmax(scores[:, -1], axis=-1).astype(mx.int32)
            next_token = mx.where(done, tokenizer.PAD, next_token)
            tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)
            done = done | (next_token == tokenizer.EOS)
            if done.all().item():
                break
        return tokens

# Test

if __name__ == "__main__":
    model = ElectroMini()
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"parameters: {n_params / 1e6:.1f} million")

    spec = mx.random.normal((2, N_BINS, 431))
    frame_mask = mx.ones((2, 431), dtype=mx.bool_)
    decoder_input = mx.random.randint(3, tokenizer.VOCAB_SIZE, (2, 50))
    scores = model(spec, frame_mask, decoder_input)
    print(f"input  spec {spec.shape}  tokens {decoder_input.shape}")
    print(f"output scores {scores.shape}  (one score per vocabulary token, at every position)")

    model.eval()
    written = model.generate(spec[:1], frame_mask[:1], max_tokens=10)
    print("untrained model writes:", [tokenizer.VOCAB[i] for i in written[0].tolist()])