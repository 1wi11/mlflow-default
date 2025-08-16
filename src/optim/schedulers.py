# src/optim/schedulers.py
from src.core.registries import SCHEDULERS
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

@SCHEDULERS.register("cosine")
def cosine(optimizer, T_max=50):
    return CosineAnnealingLR(optimizer, T_max=T_max)

@SCHEDULERS.register("step")
def step(optimizer, step_size=30, gamma=0.1):
    return StepLR(optimizer, step_size=step_size, gamma=gamma)
