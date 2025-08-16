# src/evaluators/seg_basic.py
from typing import List, Dict
import torch
from src.core.registries import EVALUATORS, METRICS, get

@EVALUATORS.register("seg_basic")
class SegEvaluator:
    def __init__(self, metrics: List[str], threshold: float = 0.5):
        self.metric_fns = [get(METRICS, m) for m in metrics]
        self.thr = threshold

    @torch.no_grad()
    def __call__(self, logits, targets) -> Dict[str, float]:
        probs = logits.sigmoid()
        out = {}
        for fn in self.metric_fns:
            name = next(k for k, v in METRICS.items() if v is fn)
            out[name] = fn(probs, targets, thr=self.thr)  # type: ignore
        return out
