# src/losses/dice.py
import torch, torch.nn as nn
from src.core.registries import LOSSES

@LOSSES.register("bce_dice")
class BCEDice(nn.Module):
    def __init__(self, bce_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(); self.w = bce_weight
    def forward(self, logits, targets, eps=1e-7):
        bce = self.bce(logits, targets) * self.w
        probs = torch.sigmoid(logits)
        inter = (probs*targets).sum(dim=(1,2,3))
        dice  = (2*inter+eps)/(probs.sum(dim=(1,2,3))+targets.sum(dim=(1,2,3))+eps)
        return bce + (1 - dice.mean())