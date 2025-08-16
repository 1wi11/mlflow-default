# src/core/factory.py
from typing import Any, Dict
from .registries import get, MODELS, LOSSES, METRICS, EVALUATORS, OPTIMIZERS, SCHEDULERS

def _kwargs(cfg: Dict[str, Any]):  # name 제외
    return {k: v for k, v in cfg.items() if k != "name"}

def build_model(cfg):     return get(MODELS,     cfg["name"])(**_kwargs(cfg))
def build_loss(cfg):      return get(LOSSES,     cfg["name"])(**_kwargs(cfg))
def build_optimizer(params, cfg):  return get(OPTIMIZERS, cfg["name"])(params, **_kwargs(cfg))
def build_scheduler(optim, cfg):   return get(SCHEDULERS, cfg["name"])(optim, **_kwargs(cfg))
def build_evaluator(cfg): return get(EVALUATORS, cfg["name"])(**_kwargs(cfg))
