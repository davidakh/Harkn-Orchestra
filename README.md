---
license: mit
library_name: mlx
tags:
  - music-transcription
  - guitar
  - tablature
  - audio
  - mlx
  - transformer
  - encoder-decoder
language:
  - en
---

# Harkn Orchestra

**Audio in, guitar tab out.** I built Harkn Orchestra, an 8.7M-parameter music-to-note model, from scratch. It listens to a guitar recording and writes the tab: string, fret, timing, playing technique, strums and beats.

I made the training data with my own guitar synthesizer. It composes original music and performs it on a physical model of a six-string guitar, so every note comes with its exact string and fret. Because the data never used scraped or copyrighted songs, there were no licenses to violate. The synthesizer produced 100,000 clips, about 194 hours of labeled audio, which I used to pre-train the model.

| | |
|---|---|
| Tab F1 (string + fret, ±50 ms) | **99%** |
| Pitch F1 | **97.6%** |
| Clips transcribed perfectly | **98%** |
| Parameters | **8.7M** |
| Training | ~23 hours on one Mac Studio |

*These scores come from 4,949 held-out test clips whose scores never appeared in training. Results on real recordings are covered [below](#on-real-guitar).*

## Training

I trained the model on a single Mac Studio in about 23 hours. I used mixed precision: float16 math, float32 master weights and dynamic loss scaling.

I didn't train with PyTorch, which I estimated would take about 3 days on my Mac. Instead I used Apple's [MLX](https://github.com/ml-explore/mlx) framework, which runs natively on Apple Silicon. It replaced PyTorch for training and brought the run down to hours.

The run lasted 60 epochs, about 140k steps at 8,192 tokens per batch, with learning rate 5e-4 and 1,000 warmup steps. Validation loss bottomed out at epoch 33 (0.2835), and that is the checkpoint released here. After epoch 33 the model started to overfit.

## Example

The model's output for a held-out test clip, a 7.6-second electric guitar arpeggio in E. It got all 35 notes right:

```
[0.12 s]
e|----------12----------12---------8---------8----------10----------10-|
B|-------12----------12---------10--------10---------10----------10----|
G|----12----------12---------10--------10---------10----------10-------|
D|---------------------------------------------------------------------|
A|-------------------------8---------8---------------------------------|
E|-12----------12------------------------------10----------10----------|
```

## Use it

```bash
pip install -r requirements.txt        # on Linux: pip install "mlx[cpu]" librosa soundfile
python transcribe.py my_riff.wav       # prints the tab
python transcribe.py my_riff.wav --json notes.json   # also saves every note
```

It accepts any audio format librosa can read. Long recordings are processed in 10-second windows. On Apple Silicon, add `--fp16` for speed.

From Python:

```python
import json
from transcribe import load_model, transcribe, ascii_tab

model = load_model("model.safetensors", fp16=False)
stats = json.load(open("spectrogram_stats.json"))
result = transcribe("my_riff.wav", model, stats)
print(ascii_tab(result["notes"]))
# result["notes"] -> [{"onset": 0.12, "string": 6, "fret": 12, "techniques": []}, ...]
```

## How it works

```
audio (22.05 kHz mono, ≤10 s)
  → constant-Q spectrogram: 252 bins, 3 per semitone, from D2; hop 512 (~23 ms)
  → 2 × Conv1d (kernel 3) front end
  → transformer encoder: 4 layers, d=256, 4 heads
  → transformer decoder: 4 layers, cross-attends to the audio
  → tokens, generated one at a time
  → tab
```

**Tokens.** The tab is written as a sequence drawn from a 1,053-token vocabulary:

- `time_0` … `time_1000`: 10 ms steps inside the window
- `string_1` … `string_6`, `fret_0` … `fret_24` and `fret_x` (dead note)
- 13 techniques: hammer-on, pull-off, slide, slide-out, ½/full/1½-step bends, release, vibrato, harmonic, palm mute, ghost note and let ring
- `strum_down` / `strum_up` to group chord notes, and `beat` / `downbeat`

One note looks like `time_42 tech_hammer_on string_3 fret_7`. The tokenizer round-trips all 100,000 training clips with zero lost notes.

## Training data

- **500,000 clips** of 4–10 s each, **194 hours** of audio and 4.2M notes, from 55,629 composed scores
- **Split by score**, so no test score (in any rendering) appears in training. The split is 90,079 train, 4,972 validation and 4,949 test clips.
- **Variety by design (domain randomization):** electric (36%), steel-string (28%) and nylon (16%) guitars, with randomized string physics, pickups, amp rigs, effects, hum and room sound
- **Musical content:** single notes, arpeggios, strumming, power-chord riffs and mixed parts, in 4/4, 3/4 and 6/8
- **Context:** 55% solo guitar, 25% guitar in a band mix, and 20% guitarless (the model writes a guitar part for music without a guitar)

## On real guitar

As a reality check, I ran the model on 60 short recordings of single notes from my own guitar, covering all six strings and frets 0–19. The model never trained on real audio.

- **Right pitch: 55 of 60 (92%)**
- **Exact string and fret: 20 of 60 (54%)**
- **Exactly one note written: 41 of 60.** Otherwise it usually repeated the note or added a chord.

The pitch result shows the synthetic data carries over to a real instrument. The string choice is the weak spot. A lone note can be played in several places on the neck, and without surrounding notes for context the model often picks a different, equally valid position. Closing that gap takes real recordings in training, which is the next step.

## Limitations

- **Mostly tested on synthetic audio.** The headline numbers come from synthetic test clips. Real recordings will score lower.
- **Standard tuning, guitar only.** There's no bass, piano, alternate tuning or capo.
- **10-second windows.** A note that rings across a window boundary can be cut or doubled.
- **Slow on CPU.** Generation is greedy and has no key-value cache. It runs on the Apple GPU through MLX. On a plain CPU, expect one to several minutes per 10 seconds of busy playing.

## Files

| file | what it is |
|---|---|
| `model.safetensors` | weights (float32, 35 MB) |
| `model_mlx.py` | the model |
| `tokenizer.py` | tab ↔ tokens |
| `spectrogram_stats.json` | per-bin normalization from the training set |
| `transcribe.py` | audio → tab |
| `config.json` | architecture, audio and training settings |

## About this project

This was a fun and educational project. I had built simple GPTs (text-to-text models) before, but a music-to-note model was a whole different undertaking. Creating the training data and handling the differences in training itself taught me how to build models of all kinds.

Built by **David Akhmedbayev**, a designer and Swift developer.
