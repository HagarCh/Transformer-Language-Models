from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def batch_to_labeled_samples(batch: torch.IntTensor) -> [torch.IntTensor, torch.IntTensor]:
    # The batches that we get from the reader have corpus-sequences of length max-context + 1.
    # We need to translate them to input/output examples, each of which is shorter by one.
    # That is, if our input is of dimension (b x n) our output is two tensors, each of dimension (b x n-1)
    inputs = batch[:,:-1] 
    labels = batch[:,1:] 
    return (inputs, labels)

def compute_loss(logits, gold_labels, pad_token_id=0):
    B, N, V = logits.size()
    logits_flat = logits.reshape(B*N, V)
    gold_labels_flat = gold_labels.reshape(B*N)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    loss = loss_fn(logits_flat, gold_labels_flat)
    return loss

