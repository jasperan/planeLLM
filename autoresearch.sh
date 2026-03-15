#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python - <<'PY'
import subprocess
import sys
import time

start = time.perf_counter()
subprocess.run(
    [sys.executable, "-m", "pytest", "tests", "-q"],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
total = time.perf_counter() - start
print(f"METRIC pytest_s={total:.6f}")
PY
