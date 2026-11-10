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
    if torch.backend.mps.is_available(): # checks if mac specific metal training backend is available
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
def evaluation(model, loader, loss_fn, device, max_batches: int):
    model.eval() # switches off dropout or variation so that the exam is the same every time
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for i, batch in enumerate(loader):
        if i == max_batches: # limit amount of batches for exam
            break
        batch = to_device(batch, device) 
        loss, correct, n = compute_loss(model, batch, loss_fn)
        total_loss += loss.item() * n.item() # weigh each batch by how many real tokens it included aka non-padding tokens
        total_correct += correct.item()
        total_tokens += n.items()
    model.train() # switchb dropout back on before training continues
    return total_loss / total_tokens, total_correct / total_tokens
