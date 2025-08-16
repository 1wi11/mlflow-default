# src/optim/optimizers.py
import torch
from src.core.registries import OPTIMIZERS

@OPTIMIZERS.register("adamw")
def adamw(params, lr=1e-3, weight_decay=0.01):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

@OPTIMIZERS.register("sgd")
def sgd(params, lr=1e-2, momentum=0.9, weight_decay=0.0, nesterov=False):
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
