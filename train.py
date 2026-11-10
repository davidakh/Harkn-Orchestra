"""Electro Traing: shows batches, measures mistakes, nudges params, repeats."""

import argparse # terminal options
import csv # read the dataset
import json # read the dataset
import math # for learning rate curve
import time # measuring learning speed
from pathlib import Path # a simplier way of handling paths

import torch # a deep-learning library, runs math on your GPU
from torch import nn # PyTorches premade transformer neural nets

import tokenizer # imports tokenizer.py
from dataset import make_loader # imports dataset.py
from model import Electro # imports the model from model.py

# Helpers

def pick_device() -> torch.device:
    if torch.backends.mps.is_available(): # checks if mac specific metal training backend is available
        return torch.device("mps") # uses mps if available
    if torch.cuda.is_available(): # checks if nvidia specifc cuda backend is available
        return torch.device("cuda") # uses cuda if available
    return torch.device("cpu") # if neither are available then it uses the cpu

def learning_rate(step: int, peak: float, warmup: int, total: int) -> float: # how big a nudge is at any given step
    if step < warmup: # warmup the first 1000 step and gradually increase nudges
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))) # consine delay to balance quick learning and fine-tuning

def to_device(batch: dict, device: torch.device) -> dict: # move batches onto the GPU if available
    return {name: value.to(device) for name, value in batch.items()}

# Measuring Loss

def compute_loss(model, batch, loss_fn):
    scores = model(batch["spec"], batch["frame_mask"], batch["decoder_input"]) # model's predictions for batch and position
    loss = loss_fn(scores.reshape(-1, scores.shape[-1]), batch["target"].reshape(-1)) # reshapes the clip positions for loss function and matches list of correct answer
    real = batch["target"] != tokenizer.PAD # determines which are real tokens and which are simply padding
    correct = ((scores.argmax(dim=-1) == batch["target"]) & real).sum() # counts how many correct answer there are simplier than loss to compute
    return loss, correct, real.sum()

# Validation

@torch.no_grad() # not for training
def evaluate(model, loader, loss_fn, device, max_batches: int):
    model.eval() # switches off dropout or variation so that the exam is the same every time
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for i, batch in enumerate(loader):
        if i == max_batches: # limit amount of batches for exam
            break
        batch = to_device(batch, device) 
        loss, correct, n = compute_loss(model, batch, loss_fn)
        total_loss += loss.item() * n.item() # weigh each batch by how many real tokens it included aka non-padding tokens
        total_correct += correct.item()
        total_tokens += n.item()
    model.train() # switchb dropout back on before training continues
    return total_loss / total_tokens, total_correct / total_tokens

# Preview 

@torch.no_grad()
def preview(model, dataset, device, n_tokens: int = 40):
    model.eval()
    spec, tokens = dataset[0]
    spec = spec[None].to(device)
    frame_mask = torch.ones(1, spec.shape[2], dtype=torch.bool, device=device)
    written = model.generate(spec, frame_mask, max_tokens=n_tokens)[0].tolist()
    model.train()
    print("  truth :", " ".join(tokenizer.decode(tokens.tolist())[: n_tokens - 1]))
    print("  model :", " ".join(tokenizer.decode(written)))

# Saving and Loading

def save_checkpoint(path: Path, model, optimizer, next_epoch: int, step: int, best_val: float): # saves snapshots so progress is not lost
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": model.config, "next_epoch": next_epoch, "step": step, "best_val": best_val}, path) # saves learned numbers, optimizer's memory, the models settings, the next epoch, the next step, and best evaluation so it can continue training

# Training Loop

def main():
    p = argparse.ArgumentParser(description="Train Electro")
    p.add_argument("--data", required=True, help="prepared dataset folder")
    p.add_argument("--run", required=True, help="folder for checkpoints and the log, e.g. runs/mini_v1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4, help="peak learning rate")
    p.add_argument("--warmup", type=int, default=1000, help="steps to ramp the learning rate up")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-batches", type=int, default=50, help="how many validation batches per exam")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=0, help="stop after N steps (for a speed test)")
    p.add_argument("--overfit", type=int, default=0, help="train on only N clips, to prove it can learn")
    p.add_argument("--resume", action="store_true", help="continue from the run's last.pt")
    args = p.parse_args()

    torch.manual_seed(0)
    device = pick_device()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)

    # The data
    batch_size = min(args.batch_size, args.overfit) if args.overfit else args.batch_size
    train_loader = make_loader(args.data, "train", batch_size=batch_size, workers=args.workers)
    val_loader = make_loader(args.data, "val", batch_size=batch_size, workers=args.workers)
    if args.overfit:
        train_loader.dataset.items = train_loader.dataset.items[: args.overfit]
        train_loader.dataset.augment = False
    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps or args.epochs * steps_per_epoch

    model = Electro().to(device) # build the model and send it to the gpu
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD) # loss without padding
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01) # nudges weights aka params, memory from original transformer paper, and anchors params to 0

    start_epoch, step, best_val = 0, 0, float("inf")
    if args.resume and (run / "last.pt").exists(): # load snapshot of restart
        saved = torch.load(run / "last.pt", map_location=device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, step, best_val = saved["next_epoch"], saved["step"], saved["best_val"]
        print(f"resumed from epoch {start_epoch}, step {step}") # prints restart specifics

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}  parameters: {n_params / 1e6:.1f} M  train clips: {len(train_loader.dataset)}")
    print(f"batch size: {batch_size}  steps per epoch: {steps_per_epoch}  total steps: {total_steps}")
    (run / "args.json").write_text(json.dumps(vars(args), indent=2)) # saves settings of the run

    model.train() # train model with variation enabled
    epoch = start_epoch # loads epoch
    running, since_log, timer = torch.zeroes((), device=device), 0, time.time() # calculate totals for progress markers
    try:
        for epoch in range(start_epoch, args.epochs): # one pass through all the training clips
            epoch_total, epoch_steps = torch.zeros((), device=device), 0

            for batch in train_loader: # one batch at a time aka one step
                lr = learning_rate(step, args.lr, args.warmup, total_steps) # steps nudge amount
                for group in optimizer.param_groups:
                    group["lr"] = lr

                # Forward Pass

                batch = to_device(batch, device) # guesses by the model
                loss, _, _ = compute_loss(model, batch, loss_fn) # score the guess

                # Backpropagation

                optimizer.zero_grad(set_to_none=True) # figures out which way to nudge weights
                loss.backward() # backpropagation
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # scales gradients to match model

                # Nudge Weights

                optimizer.step() # use the data from back step to nudge weights

                running += loss.detach() # add up the loss
                epoch_total += loss.detach()
                step += 1
                since_log += 1 
                epoch_steps += 1 

                if step % args.log_every == 0: # every 50 steps it prints the average loss, learning late, and speed
                    avg = running.item() / since_log
                    speed = since_log * batch_size / (time.time() - timer)
                    print(f"epoch {epoch}   step {step}/{total_steps}   loss {avg:.3f}  " f"lr {lr:.2e}   {speed:.0f} clips/s")
                    running, since_log, timer = torch.zeros((), device=device), 0, time.time()

                if args.max_steps and step >= args.max_steps:
                    break

                train_loss = epoch_total.item() / max(1, epoch_steps)

                if args.overfit:
                    print(f"epoch {epoch} done  train loss {train_loss:.3f}")
                else:
                    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device, args.val_batches)
                    print(f"epoch {epoch} done  train loss {train_loss:.3f}  "
                        f"val loss {val_loss:.3f}  val accuracy {val_acc:.1%}")
                    preview(model, val_loader.dataset, device)

                    new_log = not (run / "log.csv").exists()
                    with open(run / "log.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        if new_log:
                            writer.writerow(["epoch", "step", "train_loss", "val_loss", "val_accuracy"])
                        writer.writerow([epoch, step, round(train_loss, 4), round(val_loss, 4), round(val_acc, 4)])

                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(run / "best.pt", model, optimizer, epoch + 1, step, best_val)
                        print(f"  new best, saved {run / 'best.pt'}")

                save_checkpoint(run / "last.pt", model, optimizer, epoch + 1, step, best_val)

                if args.max_steps and step >= args.max_steps:
                    break

    except KeyboardInterrupt:
        save_checkpoint(run / "last.pt", model, optimizer, epoch, step, best_val)
        print(f"\nstopped. Progress saved to {run / 'last.pt'}; run again with --resume to restart epoch {epoch}.")
        return

    if args.overfit:
        print("memorization check, on a training clip:")
        preview(model, train_loader.dataset, device)
    print("done")


if __name__ == "__main__":
    main()