"""Electro Evaluate: test for the trained model."""

import argparse # terminal options
import csv # read the dataset
import json # read the dataset
import statistics # for median timing error
import time # measuring learning speed
from pathlib import Path # a simplier way of handling paths

import mlx.core as mx # imports MLX
from mlx.utils import tree_map # imports mlx utilities

import tokenizer # imports tokenizer.py
from dataset import ElectroDataset, BucketBatchSampler, bucket_for, collate # import dataset.py functions
from model_mlx import MAX_TOKENS, Electro # import model_mlx.py

OPEN_STRINGS = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64} # standard tuning

# IDs to Notes

def to_notes(ids: list[int]) -> list[dict]:
    return tokenizer.detokenize(tokenizer.decode(ids))["notes"]

def tab_key(note: dict):
    return (note["string"], note["fret"])

def pitch_key(note: dict):
    if note["fret"] is None:
        return ("dead", note["string"])
    return OPEN_STRINGS[note["string"]] + note["fret"]

# Matcher

def match(truth: list[dict], guess: list[dict], key, tolerance: float) -> list[tuple]:
    used = [False] * len(guess)
    pairs = []
    for t in truth:
        best, best_gap = None, tolerance
        for j, g in enumerate(guess):
            gap = abs(g["onset"] - t["onset"])
            if not used[j] and key(g) == key(t) and gap <= best_gap:
                best, best_gap = j, gap
        if best is not None:
            used[best] = True
            pairs.append((t, guess[best]))
    return pairs

def scores(hits: int, n_truth: int, n_guess: int) -> tuple[float, float, float]:
    precision = hits / n_guess if n_guess else 0.0
    recall = hits / n_truth if n_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

# Run

def main():
    p = argparse.ArgumentParser(description="Evaluate a trained Electro Mini (MLX) model.")
    p.add_argument("--data", required=True, help="prepared dataset folder (has manifest.csv)")
    p.add_argument("--weights", required=True, help="e.g. runs/mlx_v1/best.safetensors")
    p.add_argument("--split", default="test", help="test, val or train")
    p.add_argument("--limit", type=int, default=0, help="only the first N clips (for a quick check)")
    p.add_argument("--tolerance-ms", type=float, default=50, help="how far off a note's timing may be")
    p.add_argument("--batch-tokens", type=int, default=8192)
    p.add_argument("--fp16", action="store_true", help="run the model in float16 (faster)")
    args = p.parse_args()

    model = Electro()
    model.load_weights(args.weights)
    dtype = mx.float16 if args.fp16 else mx.float32
    model.update(tree_map(lambda w: w.astype(dtype), model.parameters()))
    model.eval()

    dataset = ElectroDataset(args.data, args.split)
    if args.limit:
        dataset.items = dataset.items[: args.limit]
    sampler = BucketBatchSampler(dataset, args.batch_tokens, seed=0)
    tolerance = args.tolerance_ms / 1000

    rows, timing_errors = [], []
    totals = {"truth": 0, "guess": 0, "tab": 0, "pitch": 0, "technique": 0, "finished": 0, "perfect": 0}
    by_group = {size: {"truth": 0, "guess": 0, "tab": 0} for size in (128, 256, 512, 1024)}
    started = time.time()


    for n_batches, indices in enumerate(sampler, start=1):
        batch = collate([dataset[i] for i in indices])
        spec = mx.array(batch["spec"].numpy()).astype(dtype)
        frame_mask = mx.array(batch["frame_mask"].numpy())
        cap = min(MAX_TOKENS, 2 * batch["decoder_input"].shape[1])
        written = model.generate(spec, frame_mask, max_tokens=cap)
        mx.eval(written)

        for row, index in enumerate(indices):
            item = dataset.items[index]
            guess_ids = written[row].tolist()
            truth = to_notes(batch["target"][row].tolist())
            guess = to_notes(guess_ids)
            tab_pairs = match(truth, guess, tab_key, tolerance)
            pitch_pairs = match(truth, guess, pitch_key, tolerance)
            same_technique = sum(set(t["techniques"]) == set(g["techniques"]) for t, g in tab_pairs)
            finished = tokenizer.EOS in guess_ids
            perfect = len(tab_pairs) == len(truth) == len(guess)

            totals["truth"] += len(truth)
            totals["guess"] += len(guess)
            totals["tab"] += len(tab_pairs)
            totals["pitch"] += len(pitch_pairs)
            totals["technique"] += same_technique
            totals["finished"] += finished
            totals["perfect"] += perfect
            timing_errors += [abs(g["onset"] - t["onset"]) * 1000 for t, g in tab_pairs]

            group = by_group[bucket_for(int(item["n_tokens"]) - 1)]
            group["truth"] += len(truth)
            group["guess"] += len(guess)
            group["tab"] += len(tab_pairs)

            rows.append({"clip_id": item["clip_id"], "truth_notes": len(truth), "model_notes": len(guess),
                         "tab_hits": len(tab_pairs), "pitch_hits": len(pitch_pairs),
                         "finished": finished, "perfect": perfect})

        print(f"  batch {n_batches}/{len(sampler)}  clips done: {len(rows)}  ({time.time() - started:.0f} s)")

    # Save every clip's result, next to the weights
    out = Path(args.weights).parent / f"eval_{args.split}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # The report
    n = len(rows)
    tab_p, tab_r, tab_f = scores(totals["tab"], totals["truth"], totals["guess"])
    pitch_p, pitch_r, pitch_f = scores(totals["pitch"], totals["truth"], totals["guess"])
    print()
    print(f"clips: {n}   time: {time.time() - started:.0f} s   finished with <eos>: {totals['finished'] / n:.1%}")
    print(f"notes in the real tabs: {totals['truth']}   notes the model wrote: {totals['guess']}")
    print(f"                       precision  recall   F1      (timing within {args.tolerance_ms:.0f} ms)")
    print(f"tab (string + fret)    {tab_p:8.1%}  {tab_r:6.1%}  {tab_f:6.1%}")
    print(f"pitch only             {pitch_p:8.1%}  {pitch_r:6.1%}  {pitch_f:6.1%}")
    if totals["tab"]:
        print(f"techniques right on matched notes: {totals['technique'] / totals['tab']:.1%}")
        print(f"timing error on matched notes: median {statistics.median(timing_errors):.0f} ms")
    print(f"perfect clips (every note right, nothing extra): {totals['perfect'] / n:.1%}")
    print("tab F1 by length group:")
    for size, g in by_group.items():
        if g["truth"]:
            print(f"  up to {size:4d} tokens: {scores(g['tab'], g['truth'], g['guess'])[2]:.1%}  ({g['truth']} notes)")

    summary = {"split": args.split, "clips": n, "tab_f1": tab_f, "tab_precision": tab_p, "tab_recall": tab_r,
               "pitch_f1": pitch_f, "perfect_clips": totals["perfect"] / n, "finished": totals["finished"] / n}
    (Path(args.weights).parent / f"eval_{args.split}.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()