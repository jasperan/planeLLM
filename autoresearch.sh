#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

python - <<'PY'
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

root = os.getcwd()
python = sys.executable
start = time.perf_counter()
pytest_env = dict(os.environ)
pytest_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"


def run_command(command, *, cwd=root, env=None):
    subprocess.run(command, check=True, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)


def run_api_smoke():
    api_proc = subprocess.Popen(
        [python, "api_server.py"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 15
        while True:
            try:
                with urllib.request.urlopen("http://127.0.0.1:7880/api/status", timeout=1) as response:
                    if response.status == 200:
                        return
            except Exception:
                if time.time() > deadline:
                    raise RuntimeError("api_server.py did not become healthy in time")
                time.sleep(0.25)
    finally:
        api_proc.terminate()
        try:
            api_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_proc.kill()
            api_proc.wait(timeout=5)


jobs = [
    lambda: run_command([python, "-m", "pytest", "tests", "-q"], env=pytest_env),
    lambda: run_command([python, "podcast_controller.py", "--help"]),
    lambda: run_command(["go", "build", "./..."], cwd=os.path.join(root, "planellm-tui")),
    run_api_smoke,
]

with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
    futures = [executor.submit(job) for job in jobs]
    import gradio_app
    app = gradio_app.create_interface()
    assert app is not None
    for future in as_completed(futures):
        future.result()

total = time.perf_counter() - start
print(f"METRIC devloop_s={total:.6f}")
PY
