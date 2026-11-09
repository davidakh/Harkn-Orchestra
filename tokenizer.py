
"""Electro Tokenizer: turns a clip of audio and its answer into tokens for input and back into tabs for output"""

from __future__ import annotations # backwards compatibility for earlier versions of Python

import json # allows the tokenizer to read and annotate JSON files
from collections import defaultdict # allows the creation of a directory similar to Numbers or Excel but for code
from pathlib import Path # a simplier way of handling file paths

# Tokenizer Definition

TIME_STEP = 0.01 # a token is present every 10 ms
MAX_SECONDS = 10.0 # caps the length of a singular clip the tokenizer handles at any given
N_TIME = int(round(MAX_SECONDS / TIME_STEP)) + 1 # calculates the amount of tokens that can exist within the 10 ms which is 1001
MAX_FRET = 24 # highest fret the tokenizer can write to
STRINGS = [6, 5, 4, 3, 2, 1] # strings within their canonical order, from low to high
STRUM_GROUP_S = 0.002 # defines which notes are within the same strum or within 2 ms of each other

SPECIALS = ["<pad>", "<sos>", "<eos>"] # bookkeeping tokes: pad which fills empty space, sos which marks the start of the sequence, and eos which is the end of sequence toke
MARKERS = ["beat", "downbeat", "strum_down", "strum_up"] # beat and strum definitions 
TECHNIQUES = [
    "hammer_on", "pull_off", "slide", "slide_out",
    "bend_half", "bend_full", "bend_1.5", "release",
    "vibrato", "harmonic", "palm_mute", "ghost", "let_ring",
] # techniques that can be a part of a note token

# Token Categorization
# For context the quotes are the actual formatting of the output of the tokenizer with the {name}_{value} format.

def build_vocab() -> list[str]: # a function that builds the processed tokens
    tokens = list(SPECIALS) # starts with the special tokens that define the space, start, and end of the token sequence
    tokens += [f"time_{i}" for i in range(N_TIME)] # adds the time tokens to the sequence
    tokens += MARKERS # adds the beat, downbeat, and strum tokens
    tokens += [f"tech_{t}" for t in TECHNIQUES] # adds the techniques tokens
    tokens += [f"string_{s}" for s in STRINGS] # adds the string tokens
    tokens += [f"fret_{f}" for f in range(MAX_FRET + 1)] # adds the tokens for frets 1 through 24
    tokens += ["fret_x"] # adds the dead fret token
    return tokens # these all of the possible tokens outputted by the tokenizer, this prevents the tokenizer from outputing random tokens, and helps it actually create the tokens

# Vocab Intalizer

VOCAB = build_vocab() # the full list of a total of 1053 tokens and the tokens position is its id, also the lookup for id to text
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)} # lookup for text to id
PAD, SOS, EOS = TOKEN_TO_ID["<pad>"], TOKEN_TO_ID["<sos>"], TOKEN_TO_ID["<eos>"] # saves the special token ids under readable names not simply 1, 2, 3
VOCAB_SIZE = len(VOCAB) # 1053 possible token types

# Token Technique Definer

def bend_token(step: float) -> str: # a function that turns the bend amount into a token
    options = {0.5: "bend_half", 1.0: "bend_full", 1.5: "bend_1.5"} # bend amount and type
    nearest = min(options, key=lambda s: abs(s - step)) # round incompatible amounts to the closest option
    return options[nearest] # returns the bend token rounded to a possible option

def note_techniques(note: dict) -> list[str]: # collects every technique on a singular note in order
    found = set() # ignores duplicates
    for t in note["techniques"]: # keep the techniques that are used in the note. strum, bend, and dead are handles else where
        if t in ("hammer_on", "pull_off", "slide", "release", "vibrato",
                 "harmonic", "palm_mute", "ghost", "let_ring"):
            found.add(t)
    if "bend" in note: # uses the bend function above to add the correct formatted technique for bend
        found.add(bend_token(note["nemd"]["steps"]))
    if "slide_out" in note: # a slide away from the note at its end is also handled seperately
        found.add("slide_out")
    return [t for t in TECHNIQUES if t in found] # returns techniques in their canonical order again    

def note_tokens(note: dict) -> list[str]: # build the tokens for a singular note which colleects the technique, string, and fret values together
    toks = [f"tech_{t}" for t in note_techniques(note)] # technique tokens are first in the order
    toks.append(f"string_{note['string']}") # then the string tokens
    if "dead" in note["techniques"]:
        toks.append("fret_x") # dead notes get fret_x
    else:  
        toks.append(f"fret_{note['fret']}") # otherwise its a real fret
    return toks # returns a singular notes tokens

# Strum Catcher

def time_index(t: float, start: float) -> int: # converts time in seconds to a time token
    return int(round((t - start) / TIME_STEP)) # time since the window began divided by 10, rounded. 