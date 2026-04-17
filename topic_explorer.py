"""Topic exploration and educational content generation for planeLLM."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional
import argparse
import os
import re
import time

from plane_llm_utils import (
    build_chat_details,
    build_genai_client,
    explain_oci_error,
    extract_chat_text,
    load_yaml_config,
    timestamp_slug,
)


_QUESTION_EXPLORATION_PROMPT = """As an expert educator, provide a detailed, engaging response to this question:
        {question}

        Your response should:
        1. Be detailed and educational (aim for 500-700 words)
        2. Start with a clear, direct answer to the question
        3. Include specific examples, data, or evidence to support key points
        4. Share interesting anecdotes or lesser-known facts that make the content memorable
        5. Use analogies or comparisons to explain complex concepts when appropriate
        6. Connect the topic to broader contexts or real-world applications
        7. Use a conversational, accessible tone while maintaining educational value
        8. End with a brief summary or takeaway that reinforces the main points

        Focus on accuracy, clarity, and making the content engaging for learners.
        """


class RateLimiter:
    """Simple rolling-window rate limiter."""

    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.interval = 60.0
        self.requests: list[float] = []
        self.lock = Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.requests = [req_time for req_time in self.requests if now - req_time <= self.interval]
                if len(self.requests) < self.max_requests:
                    self.requests.append(now)
                    return
                sleep_time = max(0.0, self.requests[0] + self.interval - now)
            time.sleep(sleep_time)


class TopicExplorer:
    """Generate educational questions and content about a topic."""

    def __init__(
        self,
        config_file: str = "config.yaml",
        max_workers: int = 10,
        *,
        config_data: Optional[dict[str, Any]] = None,
        genai_client: Any = None,
        oci_config: Optional[dict[str, Any]] = None,
        verbose: bool = False,
    ) -> None:
        self.config_data = config_data or load_yaml_config(config_file)
        self.compartment_id: str = self.config_data["compartment_id"]
        self.model_id: str = self.config_data["model_id"]
        self.genai_client = genai_client or build_genai_client(self.config_data, oci_config=oci_config)
        self.max_workers = max_workers
        self.verbose = verbose
        self.rate_limiter = RateLimiter(max_requests_per_minute=60)
        self.response_lock = Lock()
        self.execution_times: Dict[str, Any] = {
            "questions_generation": 0.0,
            "responses": {},
        }
        self._question_cache: dict[str, list[str]] = {}
        self._bundle_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _topic_key(topic: str) -> str:
        return topic.strip().casefold()

    def _log(self, message: str):
        if self.verbose:
            print(message)

    @staticmethod
    def _dedupe_questions(questions: List[str]) -> List[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for question in questions:
            cleaned = re.sub(r"^\d+[\).:-]?\s*", "", question).strip()
            if not cleaned or "?" not in cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            deduped.append(cleaned)
        return deduped

    def _generate_question_batch(self, topic: str, batch_id: int) -> List[str]:
        prompt = f"""You are an expert researcher and educator. Generate 3-4 specific, detailed questions about {topic}.
        The questions should:
        1. Cover different aspects of the topic (history, key concepts, applications, impact, etc.)
        2. Include both well-known fundamentals and lesser-known interesting facts
        3. Be specific enough to generate detailed, educational responses
        4. Avoid overly broad questions that can't be answered thoroughly

        For batch {batch_id}, focus on {'foundational concepts' if batch_id == 1 else 'advanced concepts' if batch_id == 2 else 'real-world applications and impact'}.

        Format: Return only the questions, one per line, without numbering."""
        response = self._call_llm(prompt)
        return self._dedupe_questions(response.splitlines())

    def _generate_questions_with_executor(self, topic: str, executor: ThreadPoolExecutor) -> list[str]:
        all_questions: list[str] = []
        future_map = {
            executor.submit(self._generate_question_batch, topic, batch_id): batch_id
            for batch_id in range(1, 4)
        }
        ordered_batches: dict[int, list[str]] = {}
        for future in as_completed(future_map):
            ordered_batches[future_map[future]] = future.result()

        for batch_id in sorted(ordered_batches):
            all_questions.extend(ordered_batches[batch_id])

        return self._dedupe_questions(all_questions)[:10]

    def generate_questions(self, topic: str) -> List[str]:
        topic_key = self._topic_key(topic)
        if topic_key in self._question_cache:
            return list(self._question_cache[topic_key])

        self._log(f"\nGenerating questions about '{topic}'...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=3) as executor:
            questions = self._generate_questions_with_executor(topic, executor)

        if not questions:
            raise RuntimeError(f"Failed to generate questions for topic: {topic}")

        duration = time.time() - start_time
        self.execution_times["questions_generation"] = duration
        self._question_cache[topic_key] = list(questions)
        self._log(f"Generated {len(questions)} questions in {duration:.2f} seconds")
        return questions

    def _call_llm(self, prompt: str) -> str:
        self.rate_limiter.acquire()
        return self._make_llm_call(prompt)

    def _make_llm_call(self, prompt: str) -> str:
        chat_detail = build_chat_details(
            prompt=prompt,
            model_id=self.model_id,
            compartment_id=self.compartment_id,
            max_tokens=3850,
            temperature=0.5,
        )

        last_error = None
        for attempt in range(2):
            try:
                response = self.genai_client.chat(chat_detail)
                return extract_chat_text(response)
            except Exception as exc:  # pragma: no cover - branch validated by tests
                last_error = exc
                if attempt == 0:
                    self._log(f"Error calling LLM, retrying once: {explain_oci_error(exc, model_id=self.model_id)}")
                    time.sleep(2)

        raise RuntimeError(f"Failed to call OCI GenAI service: {explain_oci_error(last_error, model_id=self.model_id)}") from last_error

    def _explore_question_thread(self, question: str, results: Dict[str, str]):
        start_time = time.time()
        response = self._call_llm(_QUESTION_EXPLORATION_PROMPT.format(question=question))

        with self.response_lock:
            results[question] = response
            self.execution_times["responses"][question] = time.time() - start_time

    def _render_bundle(self, topic: str, questions: List[str], results: Dict[str, str]) -> dict[str, Any]:
        full_content = f"# {topic}\n\n"
        questions_text = f"# Questions for {topic}\n\n"
        for index, question in enumerate(questions, 1):
            response = results[question]
            full_content += f"## {question}\n\n{response}\n\n"
            questions_text += f"{index}. {question}\n"
        return {
            "topic": topic,
            "questions": list(questions),
            "questions_text": questions_text,
            "content": full_content,
        }

    def _save_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        os.makedirs("./resources", exist_ok=True)
        timestamp = timestamp_slug()
        questions_file = f"questions_{timestamp}.txt"
        content_file = f"content_{timestamp}.txt"
        with open(f"./resources/{questions_file}", "w", encoding="utf-8") as handle:
            handle.write(bundle["questions_text"])
        with open(f"./resources/{content_file}", "w", encoding="utf-8") as handle:
            handle.write(bundle["content"])
        saved_bundle = dict(bundle)
        saved_bundle["questions_file"] = questions_file
        saved_bundle["content_file"] = content_file
        return saved_bundle

    def generate_topic_bundle(self, topic: str, save: bool = True) -> dict[str, Any]:
        topic_key = self._topic_key(topic)
        cached_bundle = self._bundle_cache.get(topic_key)
        if cached_bundle is not None:
            return self._save_bundle(cached_bundle) if save else dict(cached_bundle)

        total_start_time = time.time()
        self.execution_times["responses"] = {}
        results: Dict[str, str] = {}
        cached_questions = self._question_cache.get(topic_key)

        if cached_questions is not None:
            questions = list(cached_questions)
            self.execution_times["questions_generation"] = 0.0
            self._log(f"\nUsing cached questions for '{topic}'...")
            self._log(f"Exploring {len(questions)} questions in parallel...")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._explore_question_thread, question, results) for question in questions]
                for index, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    self._log(f"Completed {index}/{len(futures)} questions")
        else:
            with ThreadPoolExecutor(max_workers=max(self.max_workers, 3)) as executor:
                questions = self._generate_questions_with_executor(topic, executor)
                if not questions:
                    raise RuntimeError(f"Failed to generate questions for topic: {topic}")
                self.execution_times["questions_generation"] = time.time() - total_start_time
                self._question_cache[topic_key] = list(questions)

                self._log(f"Exploring {len(questions)} questions in parallel...")
                futures = [executor.submit(self._explore_question_thread, question, results) for question in questions]
                for index, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    self._log(f"Completed {index}/{len(futures)} questions")

        bundle = self._render_bundle(topic, questions, results)
        total_time = time.time() - total_start_time
        self._log("\n" + self._generate_timing_summary(total_time))
        self._bundle_cache[topic_key] = dict(bundle)
        return self._save_bundle(bundle) if save else bundle

    def generate_full_content(self, topic: str) -> str:
        return self.generate_topic_bundle(topic)["content"]

    def _generate_timing_summary(self, total_time: float) -> str:
        summary = ["=== Execution Time Summary ==="]
        summary.append(f"Questions Generation: {self.execution_times['questions_generation']:.2f} seconds")
        if self.execution_times["responses"]:
            summary.append("\nIndividual Question Times:")
            for question, time_taken in self.execution_times["responses"].items():
                summary.append(f"- {question[:50]}...: {time_taken:.2f} seconds")
            avg_response_time = sum(self.execution_times["responses"].values()) / len(self.execution_times["responses"])
            summary.append(f"\nAverage Response Time: {avg_response_time:.2f} seconds")
        summary.append(f"Total Execution Time: {total_time:.2f} seconds")
        return "\n".join(summary)

    def explore_question(self, question: str) -> str:
        self._log(f"\nExploring: {question}")
        start_time = time.time()
        response = self._call_llm(_QUESTION_EXPLORATION_PROMPT.format(question=question))
        duration = time.time() - start_time
        self._log(f"Generated {len(response.split())} words in {duration:.2f} seconds")
        return response


def count_tokens(text: str) -> int:
    import tiktoken

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate topic content with planeLLM")
    parser.add_argument("topic", help="Topic to explore")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    print(TopicExplorer(config_file=args.config, verbose=True).generate_full_content(args.topic))
