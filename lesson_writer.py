"""Podcast transcript generation for planeLLM."""

from __future__ import annotations

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


TRANSCRIPT_PRESETS = {
    "short": {"max_tokens": 2500, "two_speakers": "6-10", "three_speakers": "10-14"},
    "medium": {"max_tokens": 4000, "two_speakers": "10-15", "three_speakers": "15-20"},
    "long": {"max_tokens": 5500, "two_speakers": "18-24", "three_speakers": "24-30"},
}


class PodcastWriter:
    """Transform educational content into podcast transcripts."""

    def __init__(
        self,
        config_file: str = "config.yaml",
        speakers: int = 2,
        transcript_length: str = "medium",
        *,
        config_data: Optional[dict[str, Any]] = None,
        genai_client: Any = None,
        oci_config: Optional[dict[str, Any]] = None,
        verbose: bool = False,
    ) -> None:
        if speakers not in [2, 3]:
            raise ValueError("Number of speakers must be 2 or 3")
        if transcript_length not in TRANSCRIPT_PRESETS:
            raise ValueError("transcript_length must be one of: short, medium, long")

        self.config_data = config_data or load_yaml_config(config_file)
        self.compartment_id: str = self.config_data["compartment_id"]
        self.model_id: str = self.config_data["model_id"]
        self.genai_client = genai_client or build_genai_client(self.config_data, oci_config=oci_config)
        self.speakers = speakers
        self.transcript_length = transcript_length
        self.verbose = verbose

        preset = TRANSCRIPT_PRESETS[transcript_length]
        self.max_tokens = preset["max_tokens"]
        self.exchange_count_2speakers = preset["two_speakers"]
        self.exchange_count_3speakers = preset["three_speakers"]
        self.system_prompt = self._build_system_prompt()
        self.execution_times = {
            "start_time": 0.0,
            "total_time": 0.0,
            "llm_calls": [],
        }

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def _build_system_prompt(self) -> str:
        if self.speakers == 2:
            return f"""You are an expert podcast writer. Transform the following educational content into an engaging, conversational podcast transcript between two speakers:

Speaker 1: An expert educator who explains concepts clearly and engagingly
Speaker 2: A curious student who asks thoughtful questions and occasionally summarizes key points

Guidelines:
1. Start with a brief introduction by Speaker 1 welcoming listeners and introducing the topic
2. Organize the content into a logical flow of conversation
3. Include thoughtful questions from Speaker 2 that help transition between subtopics
4. Ensure Speaker 1 explains concepts in an accessible, conversational way with detailed examples
5. Include occasional moments where Speaker 2 summarizes or reflects on what they've learned
6. Dive deep into each concept with thorough explanations and real-world applications
7. Include analogies, examples, and stories to make complex ideas more accessible
8. End with a comprehensive conclusion summarizing key takeaways
9. Keep the tone educational but conversational and engaging
10. Format the transcript with clear speaker labels (Speaker 1: and Speaker 2:)
11. Aim for approximately {self.exchange_count_2speakers} exchanges between speakers
12. Match a {self.transcript_length} transcript length while staying natural and informative

Here's the educational content to transform:

"""

        return f"""You are an expert podcast writer who transforms educational content into engaging, conversational podcast transcripts. Your task is to create a natural, interactive dialogue between speakers that educates listeners on the given topic.

Speaker 1: An expert educator who explains concepts clearly
Speaker 2: A curious student who asks thoughtful questions
Speaker 3: A secondary expert who adds additional context, examples, and occasionally friendly debate

Guidelines:
1. Start with a brief introduction by Speaker 1 welcoming listeners and introducing the topic
2. Organize the content into a logical flow of conversation
3. Include thoughtful questions from Speaker 2 that help transition between subtopics
4. Ensure Speakers 1 and 3 explain concepts in an accessible, conversational way with detailed examples
5. Include occasional friendly disagreement or different perspectives between Speakers 1 and 3
6. Dive deep into each concept with thorough explanations and real-world applications
7. Include analogies, examples, and stories to make complex ideas more accessible
8. End with a comprehensive conclusion where all three speakers share final thoughts
9. Keep the tone educational but conversational and engaging
10. Format the transcript with clear speaker labels (Speaker 1:, Speaker 2:, and Speaker 3:)
11. Aim for approximately {self.exchange_count_3speakers} total exchanges between speakers
12. Match a {self.transcript_length} transcript length while staying natural and informative
13. Make the conversation feel natural by including occasional [laughs], [sighs], or [pauses]

Here's the educational content to transform:

"""

    def _call_llm(self, prompt: str) -> str:
        start_time = time.time()
        chat_detail = build_chat_details(
            prompt=prompt,
            model_id=self.model_id,
            compartment_id=self.compartment_id,
            max_tokens=self.max_tokens,
            temperature=0.7,
        )

        try:
            response = self.genai_client.chat(chat_detail)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to call OCI GenAI service: {explain_oci_error(exc, model_id=self.model_id)}"
            ) from exc
        result = extract_chat_text(response)
        duration = time.time() - start_time
        self.execution_times["llm_calls"].append(
            {
                "prompt_length": len(prompt),
                "response_length": len(result),
                "duration": duration,
            }
        )
        return result

    @staticmethod
    def _topic_title(content: str) -> str:
        match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
        return content.split("\n\n", 1)[0].lstrip("# ").strip() or "this topic"

    def create_podcast_transcript(self, input_content: str) -> str:
        self._log("\nTransforming content into podcast format...")
        self._log(f"Input content length: {len(input_content)} characters")
        self.execution_times["start_time"] = time.time()
        transcript = self._call_llm(self.system_prompt + input_content)
        self.execution_times["total_time"] = time.time() - self.execution_times["start_time"]

        timestamp = timestamp_slug()
        transcript_file = f"podcast_transcript_{timestamp}.txt"
        self._log(f"Saving transcript to file: {transcript_file}...")
        os.makedirs("./resources", exist_ok=True)
        with open(f"./resources/{transcript_file}", "w", encoding="utf-8") as file:
            file.write(transcript)
        return transcript

    def extract_questions_from_content(self, content: str) -> List[str]:
        sections = self.extract_sections_from_content(content)
        if sections:
            return [title for title, _ in sections]

        questions_section_match = re.search(r"# Questions for .+\n\n((?:\d+\.\s+.+\n?)+)", content)
        if questions_section_match:
            return re.findall(r"\d+\.\s+(.+)", questions_section_match.group(1))

        headers = re.findall(r"#+\s+(.+)", content)
        if len(headers) > 1:
            return headers[1:]

        question_sentences = re.findall(r"([^.!?]+\?)", content)
        if question_sentences:
            return question_sentences[:10]

        paragraphs = [paragraph.strip() for paragraph in content.split("\n\n") if paragraph.strip()]
        if len(paragraphs) >= 3:
            return paragraphs[1:min(6, len(paragraphs))]

        return ["Tell me about this topic"]

    def extract_sections_from_content(self, content: str) -> List[tuple[str, str]]:
        matches = list(re.finditer(r"^(#+)\s+(.+)$", content, flags=re.MULTILINE))
        if not matches:
            return []

        sections: list[tuple[str, str]] = []
        for index, match in enumerate(matches):
            title = match.group(2).strip()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            body = content[start:end].strip()

            if index == 0 and len(matches) > 1 and not title.endswith("?"):
                continue
            if title.lower().startswith("questions for"):
                continue
            if body:
                sections.append((title, body))

        return sections

    def create_detailed_podcast_transcript(self, input_content: str) -> str:
        self._log("\nCreating detailed podcast transcript by processing questions individually...")
        self._log(f"Input content length: {len(input_content)} characters")
        self.execution_times["start_time"] = time.time()

        topic_title = self._topic_title(input_content)
        sections = self.extract_sections_from_content(input_content)
        if sections:
            question_titles = [title for title, _ in sections]
        else:
            question_titles = self.extract_questions_from_content(input_content)
            sections = [(question, input_content) for question in question_titles]

        self._log(f"Extracted {len(question_titles)} questions/sections to process")

        intro_prompt = f"""You are an expert podcast writer. Create ONLY the introduction section for a podcast about the following topic:

{topic_title}

Guidelines:
1. Write only the introduction (first 2-3 exchanges)
2. Include Speaker 1 welcoming listeners and introducing the topic
3. Include Speaker 2 expressing interest and asking an initial question
4. Format with Speaker 1: and Speaker 2: labels
5. Keep it brief (about 150-200 words) and engaging
6. Do NOT start exploring the topic in depth yet
7. Make the conversation feel natural by including enthusiasm, curiosity, and light conversational markers
"""
        introduction = self._call_llm(intro_prompt)

        segments = [introduction]
        for index, (question, section_body) in enumerate(sections, start=1):
            self._log(f"Processing question {index}/{len(sections)}: {question[:50]}...")
            section_context = f"{question}\n\n{section_body}".strip()
            segment_prompt = f"""You are an expert podcast writer. Create a segment of a podcast conversation about the following question:

{question}

Use the following educational content as reference:
{section_context}

Guidelines:
1. Write only the segment discussing this specific question/topic
2. Start with Speaker 2 asking about this topic
3. Have Speaker 1 provide a detailed, thorough explanation
4. Include follow-up questions from Speaker 2 to explore the topic deeply
5. Format with Speaker 1: and Speaker 2: labels
6. Make the explanation detailed with examples, analogies, and real-world applications
7. Aim for about 400-600 words for this segment
8. Do NOT include an introduction or conclusion to the whole podcast
9. Make the conversation feel natural with occasional light reactions or pauses
"""
            segments.append(self._call_llm(segment_prompt))

        conclusion_prompt = f"""You are an expert podcast writer. Create ONLY the conclusion section for a podcast about the following topic:

{topic_title}

The podcast has covered these main points:
{', '.join(question_titles[:5])}

Guidelines:
1. Write only the conclusion (last 2-3 exchanges)
2. Include Speaker 1 summarizing key takeaways
3. Include Speaker 2 reflecting on what they've learned
4. Include a sign-off and thank you to listeners
5. Format with Speaker 1: and Speaker 2: labels
6. Keep it concise (about 150-200 words) and impactful
"""
        segments.append(self._call_llm(conclusion_prompt))

        full_transcript = "\n\n".join(segments)
        full_transcript = re.sub(r"(Speaker \d+:)\s+\1", r"\1", full_transcript)
        self.execution_times["total_time"] = time.time() - self.execution_times["start_time"]

        timestamp = timestamp_slug()
        transcript_file = f"podcast_transcript_detailed_{timestamp}.txt"
        self._log(f"Saving transcript to file: {transcript_file}...")
        os.makedirs("./resources", exist_ok=True)
        with open(f"./resources/{transcript_file}", "w", encoding="utf-8") as file:
            file.write(full_transcript)
        return full_transcript

    def _generate_timing_summary(self) -> str:
        summary = ["=== Execution Time Summary ==="]
        total_calls = len(self.execution_times["llm_calls"])
        total_prompt_length = sum(call["prompt_length"] for call in self.execution_times["llm_calls"])
        total_response_length = sum(call["response_length"] for call in self.execution_times["llm_calls"])
        total_llm_time = sum(call["duration"] for call in self.execution_times["llm_calls"])
        summary.append("\nLLM Statistics:")
        summary.append(f"  - Total LLM Calls: {total_calls}")
        summary.append(f"  - Total Prompt Length: {total_prompt_length} characters")
        summary.append(f"  - Total Response Length: {total_response_length} characters")
        summary.append(f"  - Total LLM Processing Time: {total_llm_time:.2f} seconds")
        summary.append(f"Total Execution Time: {self.execution_times['total_time']:.2f} seconds")
        return "\n".join(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create podcast transcripts with planeLLM")
    parser.add_argument("content_file", help="Path to the content file to transform")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--detailed", action="store_true", help="Create a detailed transcript")
    args = parser.parse_args()
    with open(args.content_file, "r", encoding="utf-8") as handle:
        content = handle.read()
    writer = PodcastWriter(config_file=args.config, verbose=True)
    result = writer.create_detailed_podcast_transcript(content) if args.detailed else writer.create_podcast_transcript(content)
    print(result)
