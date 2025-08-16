# src/metrics/segmentation.py
import torch
from src.core.registries import METRICS

@METRICS.register("dice")
def dice_metric(probs: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> float:
    pred = (probs > thr).float()
    inter = (pred*targets).sum()
    return float((2*inter+1e-7)/(pred.sum()+targets.sum()+1e-7))

@METRICS.register("iou")
def iou_metric(probs: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> float:
    pred = (probs > thr).float()
    inter = (pred*targets).sum()
    union = pred.sum()+targets.sum()-inter
    return float((inter+1e-7)/(union+1e-7))
