# Autoresearch: planeLLM whole-project devloop

## Objective
Reduce the wall-clock time for a reproducible whole-project verification loop while keeping the project behavior intact. The workload covers Python tests, CLI import/help, Gradio UI construction, FastAPI startup + health check, and Go TUI build.

## Metrics
- **Primary**: devloop_s (s, lower is better)
- **Secondary**: pytest_s, api_startup_s, import_weight observations, file count churn

## How to Run
`./autoresearch.sh` — prints `METRIC devloop_s=<seconds>`.

## Files in Scope
- `topic_explorer.py` — topic generation pipeline, caching, parallelism
- `lesson_writer.py` — transcript generation and prompt construction
- `tts_generator.py` — heavy import path, optional TTS backend initialization
- `api_server.py` — FastAPI startup path and health checks
- `gradio_app.py` — UI construction and lazy initialization
- `podcast_controller.py` — CLI orchestration path
- `plane_llm_utils.py` — shared helpers
- `tests/` — verification coverage and test runtime
- `requirements*.txt`, `setup.py`, `install.sh`, `README.md` — packaging and dependency layout
- `planellm-tui/` — Go TUI build path if needed

## Off Limits
- `config.yaml`
- `resources/`
- `voices/`
- OCI credentials, API keys, or any benchmark-cheating shortcuts
- Disabling tests, smoke checks, or the Go build just to look faster

## Constraints
- `python -m pytest tests -q` must stay green
- `python podcast_controller.py --help` must keep working
- `gradio_app.create_interface()` must still build
- `api_server.py` must still answer `/api/status`
- `go build ./...` in `planellm-tui/` must stay green
- No benchmark cheating, no removing work from the devloop without documenting and justifying it

## What's Been Tried
- Pre-autoresearch hardening pass completed:
  - Added safe resource path validation in FastAPI
  - Removed global Fish reference env mutation in request handling
  - Added constructor injection hooks for OCI/TTS-heavy classes
  - Added caching in `TopicExplorer`
  - Shrank detailed transcript prompts to section-local context
  - Split core vs optional local TTS dependencies
  - Repaired and expanded the Python test suite
- Confirmed wins:
  - Lazy-loaded heavyweight TTS dependencies, especially `torch`, so tests and CLI/help paths no longer pay import cost up front
  - Deferred CLI/backend imports until after argument parsing so `podcast_controller.py --help` is much cheaper
  - Disabled unrelated pytest plugin autoload in the benchmark path, which cut test startup overhead substantially without changing the test list
  - Quieted library-level progress printing in `TopicExplorer` and `PodcastWriter`
- Dead end so far:
  - Lazily loading `pydub.AudioSegment` and `tqdm` did not help the whole devloop metric
- Current bottlenecks:
  - `python -m pytest tests -q` still dominates the first half of the loop
  - Gradio import time dominates the post-pytest parallel smoke phase
