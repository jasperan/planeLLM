"""Repo-local pytest launcher.

This wrapper makes `python -m pytest ...` deterministic for this repo by
turning off unrelated third-party plugin autoload before delegating to the
real installed pytest package.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path


os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
sys.dont_write_bytecode = True
_THIS_DIR = Path(__file__).resolve().parent
_SEARCH_PATH = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != _THIS_DIR
]
_SPEC = importlib.machinery.PathFinder.find_spec("pytest", _SEARCH_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Unable to locate the installed pytest package")

_REAL_PYTEST = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("_plane_real_pytest", _REAL_PYTEST)
_SPEC.loader.exec_module(_REAL_PYTEST)

globals().update(_REAL_PYTEST.__dict__)


if __name__ == "__main__":
    raise SystemExit(_REAL_PYTEST.console_main())
