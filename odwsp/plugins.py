"""
Safe dynamic loader for locally compiled CUDA C++ / TensorRT IPluginV3 shared libraries.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSRC_DIR = REPO_ROOT / "csrc"


def load_compiled_trt_plugins(plugin_dir: Path | None = None) -> list[str]:
    """
    Loads locally compiled `libwsp_*.so` TensorRT IPluginV3 shared libraries from `csrc/`.
    Validates canonical paths and refuses symlink traversal outside the target directory.
    """
    base = (plugin_dir or DEFAULT_CSRC_DIR).resolve()
    if not base.is_dir():
        return []

    loaded: list[str] = []
    for lib_name in (
        "libwsp_portable_codesign.so",
        "libwsp_sm120_codesign.so",
    ):
        cand = (base / lib_name).resolve()
        if cand.parent != base or not cand.is_file():
            continue
        ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
        loaded.append(str(cand))
    return loaded
