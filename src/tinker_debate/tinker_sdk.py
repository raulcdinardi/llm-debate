from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def _repo_root() -> Path:
    # src/tinker_debate/tinker_sdk.py -> src -> repo root
    return Path(__file__).resolve().parents[2]


def _load_local_tinker_under_alias(*, alias: str = "tinker_local") -> ModuleType:
    repo = _repo_root()
    local_pkg_dir = repo / "tinker-local" / "src" / "tinker"
    init_py = local_pkg_dir / "__init__.py"
    if not init_py.exists():
        raise RuntimeError(f"Local tinker package not found at {init_py}")

    if alias in sys.modules:
        mod = sys.modules[alias]
        if not isinstance(mod, ModuleType):
            raise RuntimeError(f"sys.modules[{alias!r}] is not a module")
        return mod

    spec = importlib.util.spec_from_file_location(
        alias,
        init_py,
        submodule_search_locations=[str(local_pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to construct import spec for local tinker at {init_py}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def get_tinker() -> ModuleType:
    """
    Returns a module that provides the Tinker SDK surface used by this repo.

    Modes:
      - API mode (default): returns the real `tinker` package.
      - Local mode: returns the local backend loaded from `tinker-local/src/tinker`,
        but *under an alias module name* so it does not shadow the real SDK.

    Local mode triggers when:
      - env `TINKER_BACKEND=local`, OR
      - env `TINKER_LOCAL_BACKEND` is set (this is already required by the local backend).
    """
    if "TINKER_LOCAL_BACKEND" in os.environ and "TINKER_BACKEND" not in os.environ:
        os.environ["TINKER_BACKEND"] = "local"

    backend = os.environ.get("TINKER_BACKEND")
    if backend is not None and backend not in ("api", "local"):
        raise ValueError(f"TINKER_BACKEND must be 'api' or 'local', got {backend!r}")

    if backend == "local":
        return _load_local_tinker_under_alias()
    if backend == "api":
        return importlib.import_module("tinker")

    if "TINKER_LOCAL_BACKEND" in os.environ:
        return _load_local_tinker_under_alias()

    return importlib.import_module("tinker")


tinker = get_tinker()
