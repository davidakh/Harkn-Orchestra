
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

# Organizer

def time_index(t: float, start: float) -> int: # converts time in seconds to a time token
    return int(round((t - start) / TIME_STEP)) # time since the window began divided by 10, rounded. 

def group_strums(strum_notes: list) -> list: # find which notes belong to the same strum
    groups = [] # each strum includes its direction and notes
    for direction, nominal, note in sorted(strum_notes, key=lambda m: (m[0], m[1])): # consume notes based on direction and time
        last = groups[-1] if groups else None # the current group or strum
        if (last and last[0] == direction # join the group only based of the direction
                and nominal - last[1][0][1] <= STRUM_GROUP_S # checks if the notes are within 2 ms of each other
                and note["string"] not in {m[2]["string"] for m in last[1]}): # makes sure that the string is not already in the group
            last[1].append((direction, nominal, note)) # either adds it to the current group or creates a new one
        else:
            groups.append((direction, [(direction, nominal, note)])) # if note is not in a strum group then its left alone
    return groups # returns the notes either in groups of not

def tokenize(clip: dict, start: float = 0.0, end: float | None = None) -> list[str]: # allows the tokenizer to tokenize 10ms part of the input
    if end is None:
        end = min(start + MAX_SECONDS, clip["duration_s"]) # if there's no end, stop at the end of the clip
    events = defaultdict(list) # creates the clips

    downbeats = {round(t, 4) for t in clip["downbeats"]} # categorizes the type of beat
    for t in clip["beats"]: # go through every beat
        if start <= t < end:
            kind = "downbeat" if round(t, 4) in downbeats else "beat" # names the beat type
            events[time_index(t, start)].append((0, 0, [kind])) # place it in the correct clip

    strum_notes = []
    for note in clip["notes"]: # go through every note within the clip
        if "strum_offset_ms" in note: # asks whether this note is part of a strum
            nominal = note["onset"] - note["strum_offset_ms"] / 1000 # figures out when the strum started
            direction = "strum_down" if "strum_down" in note["techniques"] else "strum_up" # figures out the direction of the strum
            strum_notes.append((direction, nominal, note)) # combines the direction, start, and notes
        elif start <= note["onset"] < end: # a non-strummed note is automatically added to the clip
            events[time_index(note["onset"], start)].append( # time index of the note
                (1, -note["string"], note_tokens(note))) # organizes the strings

    for direction, members in group_strums(strum_notes): # handles the strummed notes
        nominal = sum(m[1] for m in members) / len(members) # strum duration is the average of the notes' time
        if not (start <= nominal < end): # skip strums that do not fit into this category
            continue
        members.sort(key=lambda m: -m[2]["string"]) # organize the notes within the strum low e is first and high e is last. 
        toks = [direction] # writes either strum up or down and the notes string and fret
        for _, _, note in members:
            toks += note_tokens(note) # writes either strum up or down and the notes string and fret
        events[time_index(nominal, start)].append((1, -members[0][2]["string"], toks)) # combine a singular strum into a clip

    out = [] # the empty clips we'll fill
    for step in sorted(events): # visit the clips in timely order, with empty boxes skipped automatically
        out.append(f"time_{min(step, N_TIME - 1)}") # writes the clips time
        for _, _, toks in sorted(events[step], key=lambda e: (e[0], e[1])): # writes what is within the box, beat first, then notes
            out += toks
    return out # return the tokenized clips