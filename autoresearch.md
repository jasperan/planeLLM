# Autoresearch: planeLLM pytest runtime

## Objective
Reduce the wall-clock time of `python -m pytest tests -q` for `/home/ubuntu/git/personal/planeLLM` without weakening coverage or changing the tested behaviors. This is the next useful optimization target after the broader devloop benchmark became dominated by Gradio’s third-party import cost.

## Metrics
- **Primary**: `pytest_s` (s, lower is better)
- **Secondary**: `collected_tests`, `skipped_tests`, import-weight observations

## How to Run
`./autoresearch.sh` — prints `METRIC pytest_s=<seconds>`.

## Files in Scope
- `tests/` — especially modules with optional-dependency gating and heavy imports
- `topic_explorer.py`
- `lesson_writer.py`
- `tts_generator.py`
- `api_server.py`
- `podcast_controller.py`
- `gradio_app.py`
- `plane_llm_utils.py`
- `pytest.ini`

## Off Limits
- `config.yaml`
- `resources/`
- `voices/`
- Any change that removes real assertions or silently narrows the suite
- Benchmark tricks like pre-warming imports before the timer starts

## Constraints
- The benchmark command must remain `python -m pytest tests -q` semantics, with only environment/config changes that are valid for the real suite
- The benchmark should exclude unrelated third-party pytest plugin autoload, because this repo does not depend on those plugins and they add environment-specific startup noise
- Full checks still need to pass: tests, CLI help, Gradio build, API health, and Go build
- Optional test modules may skip early only when the optional dependency is genuinely absent
- Don’t turn real logic tests into mocks that no longer cover the behavior they’re supposed to protect

## What's Been Tried
- The previous devloop target dropped from about 17.32s to about 2.57s through import deferral, parallel verification overlap, and cleaner optional test skipping.
- Confirmed pure-pytest wins so far:
  - `tests/test_topic_explorer.py` stubs the retry sleep so the test measures retry logic instead of a real backoff wait.
  - `tests/test_parler_audio.py` skips before heavy imports when Parler is unavailable.
  - `api_server.py` now exposes a local TTS generator factory hook so the API tests don’t need to import the heavy audio module just to patch it.
  - `lesson_writer.py` and `topic_explorer.py` no longer import the OCI SDK at module import time.
  - `tests/test_api_server.py` now hits the extracted `api_workflow` helpers directly instead of importing the full FastAPI app just to test transcript/audio logic.
  - `tests/test_topic_explorer.py` now uses an immediate fake executor for the dedupe/cache unit coverage and splits the public-method passthrough check from the retry-exhaustion check.
  - `tests/test_parler_audio.py` moved its nonessential imports below the early module skip.
- Dead ends in this target:
  - Lazily loading more of `tts_generator.py` did not show a consistent win.
  - Removing `pytest.ini` was noisy and not consistently better.
- Current bottlenecks:
  - `tests/test_topic_explorer.py` is still the slowest single file, mostly because the retry-exhaustion path still constructs real OCI request objects.
  - The remaining runtime is now mostly ordinary pytest process startup plus a handful of import-heavy files.
