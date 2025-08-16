"""
core/errors.py

실험 실행 중 자주 발생하는 예외를 '친절한 메시지'로 변환하고,
설정 값 검증(스키마 체크), 유사 이름 제안, MLflow에 에러를 남기는 유틸을 모아둔 모듈.

주요 기능
- ConfigValidationError / BuildError / RegistryNotFound 예외 클래스
- 스키마 검증: validate_required, validate_types
- 이름 오타 시 후보 제안: suggest_name
- 안전 실행 컨텍스트: capture_run_errors (MLflow metric/tag로 실패 기록)
- 데코레이터: require_keys, enforce_types
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Type
import difflib
import contextlib
import traceback
import os

try:
    import mlflow  # 선택적 의존
except Exception:  # mlflow 미설치 환경 고려
    mlflow = None  # type: ignore


# ---------- 예외 타입 ----------

class ConfigValidationError(ValueError):
    """YAML/딕셔너리 기반 설정이 필수 키/타입을 만족하지 않을 때."""
    pass


class RegistryNotFound(KeyError):
    """레지스트리(모델/손실/옵티마/스케줄러 등)에서 name을 찾지 못했을 때."""
    pass


class BuildError(RuntimeError):
    """객체 생성(build_*) 단계에서 발생한 일반 예외 래핑."""
    pass


# ---------- 유틸 ----------

def _fmt_path(path: Sequence[str] | None) -> str:
    return ".".join(path) + ": " if path else ""


def validate_required(cfg: Mapping[str, Any], required: Iterable[str], path: Sequence[str] | None = None) -> None:
    """
    cfg에 반드시 있어야 하는 키들이 존재하는지 검사.
    """
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ConfigValidationError(f"{_fmt_path(path)}missing required keys: {missing}")


def validate_types(cfg: Mapping[str, Any], spec: Mapping[str, Tuple[Type, ...]], path: Sequence[str] | None = None) -> None:
    """
    cfg의 각 키가 기대 타입(spec)을 만족하는지 검사.
    spec 예: {"epochs": (int,), "lr": (float, int)}
    """
    wrong = []
    for k, types in spec.items():
        if k in cfg and not isinstance(cfg[k], types):
            wrong.append(f"{k} expected {types}, got {type(cfg[k]).__name__}")
    if wrong:
        raise ConfigValidationError(f"{_fmt_path(path)}type mismatch: {', '.join(wrong)}")


def suggest_name(name: str, candidates: Iterable[str], n: int = 5) -> str:
    """
    오타가 의심되는 name에 대해 유사 후보를 제안(레벤슈타인 기반).
    """
    cand = list(candidates)
    if not cand:
        return f"'{name}' not found (no registered names)."
    near = difflib.get_close_matches(name, cand, n=n, cutoff=0.0)
    hint = ", ".join(near[:n]) if near else "(no close matches)"
    return f"'{name}' not found. Did you mean: {hint} ?"


@contextlib.contextmanager
def capture_run_errors(run_name: str | None = None, log_to_mlflow: bool = True):
    """
    with 블록 내 예외를 캡처해 보기 좋은 메시지로 출력하고,
    원하면 MLflow에 실패 태그/스택트레이스를 기록.
    """
    try:
        yield
    except Exception as e:
        tb = traceback.format_exc()
        msg = f"[RUN FAILED] {e.__class__.__name__}: {e}"
        print(msg)  # 콘솔
        print(tb)

        if log_to_mlflow and mlflow is not None:
            try:
                # 활성 Run이 없다면 임시로 시작
                active = mlflow.active_run()
                if active is None:
                    mlflow.start_run(run_name=run_name or "failed_run")
                mlflow.set_tags({"run_status": "failed", "error_type": e.__class__.__name__})
                mlflow.log_text(tb, artifact_file="logs/traceback.txt")  # MLflow 2.14+ 지원
            except Exception:
                pass  # 로깅 실패는 조용히 무시
        raise  # 상위로 다시 올려서 호출자가 제어하도록


# ---------- 데코레이터 ----------

def require_keys(*keys: str):
    """
    함수 인자로 받는 딕셔너리 cfg에 필수 키가 있는지 검사.
    주로 build_* 함수에 사용.
    """
    def deco(fn):
        def wrapper(cfg: Dict[str, Any], *a, **kw):
            validate_required(cfg, keys)
            return fn(cfg, *a, **kw)
        return wrapper
    return deco


def enforce_types(**spec: Tuple[Type, ...]):
    """
    cfg의 키 타입을 강제.
    예: @enforce_types(epochs=(int,), lr=(float,int))
    """
    def deco(fn):
        def wrapper(cfg: Dict[str, Any], *a, **kw):
            validate_types(cfg, spec)
            return fn(cfg, *a, **kw)
        return wrapper
    return deco


# ---------- 레지스트리 헬퍼 ----------

@dataclass
class NameLookup:
    """레지스트리 조회 시 에러 메시지를 표준화."""
    registry_name: str
    names: Iterable[str]

    def get_or_raise(self, name: str):
        raise RegistryNotFound(f"[{self.registry_name}] " + suggest_name(name, self.names))
