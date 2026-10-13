"""Optional exo-tbccl detection. exo must import and run without it; selecting MlxTbccl without it is a clear configuration error."""

from importlib import import_module
from typing import Protocol, cast


class ExoTbcclModule(Protocol):
    def is_available(self) -> tuple[bool, str]: ...


def _import(name: str) -> object:
    return import_module(name)


def load_exo_tbccl() -> ExoTbcclModule:
    """Import exo_tbccl lazily (ImportError if absent)."""
    return cast(ExoTbcclModule, _import("exo_tbccl"))


def tbccl_available() -> tuple[bool, str]:
    try:
        module = load_exo_tbccl()
    except ImportError as e:
        return False, f"exo_tbccl is not installed ({e})"
    return module.is_available()
