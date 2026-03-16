#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

python - <<'PY'
import os
import tempfile
import time

from topic_explorer import TopicExplorer

QUESTION_BATCHES = {
    1: [
        "What is the foundation of deterministic benchmarking?",
        "How did the project architecture evolve?",
        "Why does orchestration overhead matter?",
        "What tradeoffs show up in educational pipelines?",
    ],
    2: [
        "How can concurrency help this workflow?",
        "What bottlenecks remain in the local path?",
        "Which data structures matter most here?",
        "How should we reason about scalability?",
    ],
    3: [
        "Where does this workflow help users in practice?",
        "What mistakes do teams make when optimizing?",
        "How should newcomers approach the codebase?",
        "How should we reason about scalability?",
    ],
}
ANSWER = " ".join(["Detailed explanation with examples, caveats, and practical guidance."] * 180)
LOOPS = 200


class FakeClient:
    def chat(self, chat_detail):
        prompt = chat_detail.chat_request.messages[0].content[0].text
        if "Generate 3-4 specific, detailed questions" in prompt:
            batch = 1 if "For batch 1" in prompt else 2 if "For batch 2" in prompt else 3
            text = "\n".join(QUESTION_BATCHES[batch])
        else:
            match = next(
                question
                for batch in QUESTION_BATCHES.values()
                for question in batch
                if question in prompt
            )
            text = f"{match}\n\n{ANSWER}"
        return {"chat_response": {"choices": [{"message": {"content": [{"text": text}]}}]}}


cfg = {"compartment_id": "x", "model_id": "y"}
start = time.perf_counter()
for _ in range(LOOPS):
    with tempfile.TemporaryDirectory() as tmp:
        old = os.getcwd()
        os.chdir(tmp)
        try:
            TopicExplorer(config_data=cfg, genai_client=FakeClient(), max_workers=10).generate_topic_bundle(
                "Deterministic Topic",
                save=True,
            )
        finally:
            os.chdir(old)
total = time.perf_counter() - start
print(f"METRIC bundle_ms={1000 * total / LOOPS:.6f}")
PY
