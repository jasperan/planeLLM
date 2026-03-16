# Autoresearch: planeLLM deterministic TopicExplorer bundle runtime

## Objective
Reduce the local wall-clock cost of `TopicExplorer.generate_topic_bundle(..., save=True)` on a deterministic fake-client workload that exercises real question generation orchestration, parallel answer collection, bundle rendering, and file saving. This avoids network noise and focuses on the Python-side cost of the bundle path.

## Metrics
- **Primary**: `bundle_ms` (ms, lower is better)
- **Secondary**: `loops`, `unique_questions`, `content_chars`

## How to Run
`./autoresearch.sh` — prints `METRIC bundle_ms=<ms>`.

The benchmark runs 200 fresh `TopicExplorer.generate_topic_bundle()` calls in temporary directories, using a deterministic fake client and `save=True`, then reports the average milliseconds per bundle.

## Files in Scope
- `topic_explorer.py`
- `plane_llm_utils.py`
- `tests/test_topic_explorer.py`
- `autoresearch.sh`
- `autoresearch.md`
- `autoresearch.ideas.md`

## Off Limits
- `config.yaml`
- `resources/`
- `voices/`
- Anything that special-cases the fake benchmark client or bypasses real bundle rendering/saving logic
- Benchmark-only hacks that skip work users rely on

## Constraints
- The benchmark must keep using `generate_topic_bundle(..., save=True)`
- The fake client may only replace network I/O, not the internal bundle pipeline
- Full project checks still need to pass: tests, CLI help, Gradio build, API health, and Go build
- Don’t turn this into a cache benchmark by reusing a warmed explorer instance across iterations

## What's Been Tried
- Prior targets:
  - devloop benchmark dropped from ~17.32s to ~2.57s
  - direct `python -m pytest tests -q` dropped from ~1.53s to ~0.13s via the repo-local pytest wrapper and cheaper unit tests
- Reason for pivot:
  - further pytest-only wins are now tiny and noisy
  - the next useful performance question is the actual local TopicExplorer orchestration cost
- Current code observations:
  - `generate_questions()` uses a 3-worker executor plus `as_completed()` bookkeeping, then dedupes and truncates
  - `generate_topic_bundle()` uses a shared results dict, a lock, and a second executor with `as_completed()`
  - `_render_bundle()` builds large strings with repeated concatenation
  - `_save_bundle()` writes 2 files per bundle and generates a unique timestamp slug each time
