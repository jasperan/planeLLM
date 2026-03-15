# Autoresearch: planeLLM direct pytest runtime

## Objective
Reduce the wall-clock time of the real developer command `python -m pytest tests -q` for `/home/ubuntu/git/personal/planeLLM` without weakening coverage or changing tested behavior.

## Metrics
- **Primary**: `pytest_s` (s, lower is better)
- **Secondary**: `collected_tests`, `skipped_tests`, import-weight observations

## How to Run
`./autoresearch.sh` — prints `METRIC pytest_s=<seconds>`.

## Files in Scope
- `tests/`
- `topic_explorer.py`
- `lesson_writer.py`
- `tts_generator.py`
- `api_server.py`
- `api_workflow.py`
- `plane_llm_utils.py`
- `pytest.ini`
- `pytest.py`

## Off Limits
- `config.yaml`
- `resources/`
- `voices/`
- Any change that removes real assertions or silently narrows the suite
- Benchmark tricks like pre-warming imports before the timer starts

## Constraints
- The benchmark command is the literal user command: `python -m pytest tests -q`
- Full checks still need to pass: tests, CLI help, Gradio build, API health, and Go build
- Optional test modules may skip early only when the optional dependency is genuinely absent
- Don’t turn real logic tests into mocks that no longer cover the behavior they’re supposed to protect

## What's Been Tried
- Earlier clean-plugin benchmarking already sanded down the suite itself a lot:
  - API logic tests moved into `api_workflow.py`
  - OCI imports were deferred off module import paths
  - TopicExplorer tests stopped paying real retry sleeps and real thread-pool overhead
  - Parler tests skip before heavy imports when Parler is absent
- That got the clean-plugin harness down into the ~0.24s range.
- The remaining big gap was between:
  - clean-plugin pytest wall time: about `0.23s`
  - raw developer command wall time: about `1.5s`
- Confirmed wins on the direct developer command target:
  - A repo-local `pytest.py` wrapper now makes the literal `python -m pytest tests -q` command opt out of unrelated external plugin autoload before delegating to the installed pytest package.
  - The direct API tests still use the lightweight `api_workflow.py` helpers instead of importing the whole FastAPI app.
  - `tests/test_topic_explorer.py` now injects a tiny fake `oci` module for the retry-exhaustion unit path, so it still exercises retry behavior without constructing real OCI SDK request objects.
- Current state:
  - direct pytest wall time now sits around `0.13s` to `0.25s` depending on run-to-run noise
  - the target is now close to the startup floor for this tiny unittest-style suite
- Dead ends on the direct-command target:
  - `sitecustomize.py` did not fire in this environment for `python -m pytest`
  - removing `pytest.ini` was noisy and not consistently better
  - forcing `sys.dont_write_bytecode` in the wrapper did not hold up
