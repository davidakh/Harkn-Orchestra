"""Electro Traing: shows batches, measures mistakes, nudges params, repeats."""

import argparse # terminal options
import csv # read the dataset
import json # read the dataset
import math # for learning rate curve
import time # measuring learning speed
from pathlib import Path # a simplier way of handling paths

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_map_with_path

import tokenizer # imports tokenizer.py
from dataset import make_loader # imports dataset.py
from model_mlx import Electro # imports the model from model.py

GB = 1024 ** 3 # one gigabyte in bytes

# Helpers

def learning_rate(step: int, peak: float, warmup: int, total: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def to_mlx(batch: dict, dtype) -> dict:
    return {
        "spec": mx.array(batch["spec"].numpy()).astype(dtype),
        "frame_mask": mx.array(batch["frame_mask"].numpy()),
        "decoder_input": mx.array(batch["decoder_input"].numpy().astype(np.int32)),
        "target": mx.array(batch["target"].numpy().astype(np.int32)),
    }


def memory_report() -> str:
    return (f"{mx.get_active_memory() / GB:.1f} GB in use, "
            f"{mx.get_cache_memory() / GB:.1f} GB cached, "
            f"{mx.get_peak_memory() / GB:.1f} GB peak")

# Measuring Loss

def loss_fn(model, batch, loss_scale):
    scores = model(batch["spec"], batch["frame_mask"], batch["decoder_input"]).astype(mx.float32)
    real = batch["target"] != tokenizer.PAD
    n = real.sum()
    mistakes = nn.losses.cross_entropy(scores, batch["target"], reduction="none")
    loss = (mistakes * real).sum() / n
    correct = ((scores.argmax(axis=-1) == batch["target"]) & real).sum()
    return loss * loss_scale, (loss, correct, n)

# Validation

def evaluate(model, loader, dtype, max_batches: int):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for i, batch in enumerate(loader):
        if i == max_batches:
            break
        _, (loss, correct, n) = loss_fn(model, to_mlx(batch, dtype), 1.0)
        mx.eval(loss, correct, n)
        total_loss += loss.item() * n.item()
        total_correct += correct.item()
        total_tokens += n.item()
    model.train()
    return total_loss / total_tokens, total_correct / total_tokens

# Preview 

def preview(model, dataset, dtype, n_tokens: int = 40):
    model.eval()
    spec, tokens = dataset[0]
    spec = mx.array(spec.numpy())[None].astype(dtype)
    frame_mask = mx.ones((1, spec.shape[2]), dtype=mx.bool_)
    written = model.generate(spec, frame_mask, max_tokens=n_tokens)[0].tolist()
    model.train()
    print("  truth :", " ".join(tokenizer.decode(tokens.tolist())[: n_tokens - 1]))
    print("  model :", " ".join(tokenizer.decode(written)))

# Mixed Precision

class LossScaler:
    def __init__(self, enabled: bool, scale: float = 2.0 ** 15, grow_every: int = 1000):
        self.enabled = enabled
        self.scale = scale if enabled else 1.0
        self.grow_every = grow_every
        self.good_steps = 0

    def update(self, finite: bool):
        if not self.enabled:
            return
        if finite:
            self.good_steps += 1
            if self.good_steps == self.grow_every:
                self.scale = min(self.scale * 2, 2.0 ** 24)
                self.good_steps = 0
        else:
            self.scale = max(self.scale / 2, 1.0)
            self.good_steps = 0

# Saving and Loading

def save_checkpoint(run: Path, name: str, master: dict, optimizer, info: dict):
    mx.save_safetensors(str(run / f"{name}.safetensors"), dict(tree_flatten(master)))
    if optimizer is not None:
        mx.save_safetensors(str(run / f"{name}.optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    (run / f"{name}.json").write_text(json.dumps(info, indent=2))


def load_checkpoint(run: Path, model, optimizer):
    model.load_weights(str(run / "last.safetensors"))
    master = model.parameters()
    optimizer.init(master)
    saved = mx.load(str(run / "last.optimizer.safetensors"))
    optimizer.state = tree_map_with_path(lambda path, value: saved.get(path, value), optimizer.state)
    return master, json.loads((run / "last.json").read_text())

# Training Loop

def main():
    p = argparse.ArgumentParser(description="Train Electro Mini with MLX.")
    p.add_argument("--data", required=True, help="prepared dataset folder (has manifest.csv)")
    p.add_argument("--run", required=True, help="folder for checkpoints and the log, e.g. runs/mlx_v1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-tokens", type=int, default=8192, help="tokens per batch; lower it to use less memory")
    p.add_argument("--lr", type=float, default=5e-4, help="peak learning rate")
    p.add_argument("--warmup", type=int, default=1000, help="steps to ramp the learning rate up")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--val-batches", type=int, default=50, help="how many validation batches per exam")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=0, help="stop after N steps (for a speed test)")
    p.add_argument("--overfit", type=int, default=0, help="train on only N clips, to prove it can learn")
    p.add_argument("--resume", action="store_true", help="continue from the run's last checkpoint")
    p.add_argument("--fp16", action="store_true", help="mixed precision: float16 math, float32 master weights")
    p.add_argument("--memory-limit", type=float, default=20, help="GB; MLX aims to keep its memory under this")
    p.add_argument("--cache-limit", type=float, default=2, help="GB of spare memory MLX may keep for reuse")
    p.add_argument("--checkpoint-layers", action="store_true",
                   help="recompute layers during backprop: less memory, about 30%% slower")
    args = p.parse_args()

    # Memory Limiter

    mx.set_memory_limit(int(args.memory_limit * GB))
    mx.set_cache_limit(int(args.cache_limit * GB))

    mx.random.seed(0)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    dtype = mx.float16 if args.fp16 else mx.float32

    # Loader

    train_loader = make_loader(args.data, "train", args.batch_tokens, args.workers, limit=args.overfit)
    val_loader = make_loader(args.data, "val", args.batch_tokens, args.workers)
    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps or args.epochs * steps_per_epoch

    # Model

    model = Electro(checkpoint=args.checkpoint_layers)
    master = model.parameters()
    optimizer = optim.AdamW(learning_rate=args.lr, betas=[0.9, 0.98], weight_decay=0.01)
    scaler = LossScaler(enabled=args.fp16)

    start_epoch, step, best_val = 0, 0, float("inf")
    if args.resume and (run / "last.json").exists():
        master, saved = load_checkpoint(run, model, optimizer)
        start_epoch, step, best_val = saved["next_epoch"], saved["step"], saved["best_val"]
        scaler.scale = saved["loss_scale"] if args.fp16 else 1.0
        print(f"resumed from epoch {start_epoch}, step {step}")

    model.update(tree_map(lambda w: w.astype(dtype), master))
    mx.eval(model.parameters(), master)

    n_params = sum(w.size for _, w in tree_flatten(master))
    print(f"device: {mx.default_device()}  parameters: {n_params / 1e6:.1f} M  train clips: {len(train_loader.dataset)}")
    print(f"precision: {'mixed float16' if args.fp16 else 'float32'}  tokens per batch: {args.batch_tokens}  "
          f"steps per epoch: {steps_per_epoch}  total steps: {total_steps}")
    print(f"memory limit: {args.memory_limit:.0f} GB  cache limit: {args.cache_limit:.0f} GB")
    (run / "args.json").write_text(json.dumps(vars(args), indent=2))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def info(next_epoch):
        return {"next_epoch": next_epoch, "step": step, "best_val": best_val,
                "loss_scale": scaler.scale, "precision": "fp16" if args.fp16 else "fp32"}

    model.train()
    epoch = start_epoch
    running, since_log, seen, skipped, timer = 0.0, 0, 0, 0, time.time()

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_total, epoch_steps = 0.0, 0

            for batch in train_loader:
                lr = learning_rate(step, args.lr, args.warmup, total_steps)
                optimizer.learning_rate = lr

                # Forward Pass

                batch = to_mlx(batch, dtype)
                (_, (loss, _, _)), grads = loss_and_grad(model, batch, scaler.scale)

                # Backpropagation

                grads = tree_map(lambda g: g.astype(mx.float32) / scaler.scale, grads)
                grads, grad_norm = optim.clip_grad_norm(grads, 1.0)
                mx.eval(loss, grad_norm)
                finite = math.isfinite(grad_norm.item())

                # Nudge Weights

                if finite:
                    master = optimizer.apply_gradients(grads, master)
                    model.update(tree_map(lambda w: w.astype(dtype), master))
                    mx.eval(master, model.parameters(), optimizer.state)
                else:
                    skipped += 1
                scaler.update(finite)

                running += loss.item()
                epoch_total += loss.item()
                step += 1
                since_log += 1
                seen += batch["spec"].shape[0]
                epoch_steps += 1

                if step % args.log_every == 0:
                    speed = seen / (time.time() - timer)
                    extra = f"  scale {scaler.scale:.0f}  skipped {skipped}" if args.fp16 else ""
                    print(f"epoch {epoch}  step {step}/{total_steps}  loss {running / since_log:.3f}  "
                          f"lr {lr:.2e}  {speed:.0f} clips/s  {memory_report()}{extra}")
                    running, since_log, seen, timer = 0.0, 0, 0, time.time()
                    mx.reset_peak_memory()

                if args.max_steps and step >= args.max_steps:
                    break

            train_loss = epoch_total / max(1, epoch_steps)

            if args.overfit:
                print(f"epoch {epoch} done  train loss {train_loss:.3f}")
            else:
                val_loss, val_acc = evaluate(model, val_loader, dtype, args.val_batches)
                print(f"epoch {epoch} done  train loss {train_loss:.3f}  "
                      f"val loss {val_loss:.3f}  val accuracy {val_acc:.1%}")
                preview(model, val_loader.dataset, dtype)

                new_log = not (run / "log.csv").exists()
                with open(run / "log.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    if new_log:
                        writer.writerow(["epoch", "step", "train_loss", "val_loss", "val_accuracy"])
                    writer.writerow([epoch, step, round(train_loss, 4), round(val_loss, 4), round(val_acc, 4)])

                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(run, "best", master, None, info(epoch + 1))
                    print(f"  new best, saved {run / 'best.safetensors'}")

            save_checkpoint(run, "last", master, optimizer, info(epoch + 1))
            mx.clear_cache()

            if args.max_steps and step >= args.max_steps:
                break

    except KeyboardInterrupt:
        save_checkpoint(run, "last", master, optimizer, info(epoch))
        print(f"\nstopped. Progress saved to {run / 'last.safetensors'}; run again with --resume to restart epoch {epoch}.")
        return

    if args.overfit:
        print("memorization check, on a training clip:")
        preview(model, train_loader.dataset, dtype)
    print("done")


if __name__ == "__main__":
    main()