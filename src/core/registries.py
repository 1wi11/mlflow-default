# src/core/registries.py
from typing import Callable, Dict

class Registry(dict):
    def register(self, name: str):
        def deco(obj):
            if name in self:
                raise ValueError(f"Duplicate key in registry: {name}")
            self[name] = obj
            return obj
        return deco

MODELS     : Dict[str, Callable] = Registry()
LOSSES     : Dict[str, Callable] = Registry()
METRICS    : Dict[str, Callable] = Registry()
EVALUATORS : Dict[str, Callable] = Registry()
OPTIMIZERS : Dict[str, Callable] = Registry()
SCHEDULERS : Dict[str, Callable] = Registry()

def get(reg: Dict[str, Callable], name: str):
    if name not in reg:
        raise KeyError(f"'{name}' not found. Available: {sorted(list(reg.keys()))}")
    return reg[name]
