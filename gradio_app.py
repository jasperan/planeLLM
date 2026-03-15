#!/usr/bin/env python
"""Gradio launcher for the planeLLM workflow."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import gradio as gr

from lesson_writer import PodcastWriter
from topic_explorer import TopicExplorer
from tts_generator import TTSGenerator


os.makedirs("./resources", exist_ok=True)


class PlaneLLMInterface:
    """Stateful Gradio wrapper around the planeLLM pipeline."""

    def __init__(self):
        self.topic_explorer = None
        self.podcast_writer = None
        self.tts_generator = None
        self.update_available_files()

    def _get_topic_explorer(self) -> TopicExplorer:
        if self.topic_explorer is None:
            self.topic_explorer = TopicExplorer()
        return self.topic_explorer

    def _get_podcast_writer(self, transcript_length: str = "medium") -> PodcastWriter:
        if self.podcast_writer is None or self.podcast_writer.transcript_length != transcript_length:
            self.podcast_writer = PodcastWriter(transcript_length=transcript_length)
        return self.podcast_writer

    def update_available_files(self) -> Dict[str, List[str]]:
        resources_dir = "./resources"
        os.makedirs(resources_dir, exist_ok=True)
        all_files = sorted(os.listdir(resources_dir))
        self.available_files = {
            "content": [f for f in all_files if f.endswith(".txt") and ("content" in f or "raw_lesson" in f)],
            "questions": [f for f in all_files if f.endswith(".txt") and "questions" in f],
            "transcripts": [f for f in all_files if f.endswith(".txt") and "podcast" in f],
            "audio": [f for f in all_files if f.endswith(".mp3")],
        }
        return self.available_files

    def generate_topic_content(self, topic: str, progress=gr.Progress()) -> Tuple[str, str, str]:
        if not topic:
            return "", "", "Error: Please enter a topic"

        try:
            progress(0, desc="Initializing...")
            explorer = self._get_topic_explorer()
            progress(0.2, desc="Generating questions and content in parallel...")
            bundle = explorer.generate_topic_bundle(topic)
            progress(1.0, desc="Done!")
            self.update_available_files()
            return (
                bundle["questions_text"],
                bundle["content"],
                f"Content generated successfully and saved to {bundle['content_file']}",
            )
        except Exception as exc:
            return "", "", f"Error: {exc}"

    def create_podcast_transcript(
        self,
        content_file: str,
        detailed_transcript: bool,
        transcript_length: str,
        progress=gr.Progress(),
    ) -> Tuple[str, str]:
        if not content_file:
            return "", "Error: Please select a content file"

        try:
            progress(0, desc="Reading content file...")
            with open(f"./resources/{content_file}", "r", encoding="utf-8") as handle:
                content = handle.read()

            writer = self._get_podcast_writer(transcript_length=transcript_length)
            if detailed_transcript:
                progress(0.2, desc="Generating detailed transcript...")
                transcript = writer.create_detailed_podcast_transcript(content)
            else:
                progress(0.2, desc="Generating standard transcript...")
                transcript = writer.create_podcast_transcript(content)

            transcript_files = sorted(
                [f for f in os.listdir("./resources") if f.startswith("podcast_transcript_") and f.endswith(".txt")]
            )
            transcript_file = transcript_files[-1] if transcript_files else "podcast_transcript_latest.txt"
            progress(1.0, desc="Done!")
            self.update_available_files()
            return transcript, f"Transcript generated successfully and saved to {transcript_file}"
        except Exception as exc:
            return "", f"Error: {exc}"

    def generate_podcast_audio(
        self,
        transcript_file: str,
        model_type: str,
        speaker1_voice: str = "male_clear",
        speaker2_voice: str = "female_expressive",
        speaker3_voice: str = "male_expressive",
        fish_reference: str = "",
        fish_emotion: str = "(neutral)",
        progress=gr.Progress(),
    ) -> Tuple[str, str]:
        if not transcript_file:
            return "", "Error: Please select a transcript file"

        try:
            progress(0, desc=f"Initializing {model_type} model...")
            global TTSGenerator
            if TTSGenerator is None:
                from tts_generator import TTSGenerator as TTSGeneratorClass
                TTSGenerator = TTSGeneratorClass

            if self.tts_generator is None or self.tts_generator.model_type != model_type:
                self.tts_generator = TTSGenerator(model_type=model_type, fish_reference_id=fish_reference or None)

            if model_type == "parler" and hasattr(self.tts_generator, "set_voice_type"):
                self.tts_generator.set_voice_type("Speaker 1", speaker1_voice)
                self.tts_generator.set_voice_type("Speaker 2", speaker2_voice)
                self.tts_generator.set_voice_type("Speaker 3", speaker3_voice)

            if model_type == "fish" and fish_reference:
                self.tts_generator.fish_reference_id = fish_reference
                for speaker in self.tts_generator.fish_speaker_map:
                    self.tts_generator.fish_speaker_map[speaker] = fish_reference

            transcript_path = f"./resources/{transcript_file}"
            if not os.path.isfile(transcript_path):
                return "", f"Error: Transcript file not found at {transcript_path}"

            with open(transcript_path, "r", encoding="utf-8") as handle:
                transcript = handle.read()

            progress(0.2, desc="Generating audio...")
            result_path = self.tts_generator.generate_podcast(transcript, output_path=None)

            if isinstance(result_path, str) and result_path.endswith(".txt") and "error" in os.path.basename(result_path):
                with open(result_path, "r", encoding="utf-8") as handle:
                    return "", f"Error: {handle.readline().strip()}"

            progress(1.0, desc="Done!")
            self.update_available_files()
            return result_path, f"Podcast audio generated successfully and saved to {os.path.basename(result_path)}"
        except Exception as exc:
            return "", f"Error: {exc}"


def create_interface():
    interface = PlaneLLMInterface()

    with gr.Blocks(title="planeLLM Interface") as app:
        gr.Markdown("# planeLLM: Educational Content Generation System")

        with gr.Tabs():
            with gr.Tab("Topic Explorer"):
                gr.Markdown("## Generate Educational Content")
                topic_input = gr.Textbox(label="Topic", placeholder="Enter a topic")
                generate_button = gr.Button("Generate Content")
                questions_output = gr.Textbox(label="Generated Questions", lines=10, interactive=False)
                content_output = gr.Textbox(label="Generated Content", lines=20, interactive=False)
                status_output = gr.Textbox(label="Status", interactive=False)
                generate_button.click(
                    fn=interface.generate_topic_content,
                    inputs=[topic_input],
                    outputs=[questions_output, content_output, status_output],
                )

            with gr.Tab("Lesson Writer"):
                gr.Markdown("## Create Podcast Transcript")
                content_file_dropdown = gr.Dropdown(
                    label="Select Content File",
                    choices=interface.available_files["content"],
                    interactive=True,
                )
                refresh_content_button = gr.Button("Refresh Files")
                detailed_transcript = gr.Checkbox(
                    label="Detailed Processing",
                    value=True,
                    info="Process each question individually for a richer transcript",
                )
                transcript_length = gr.Radio(
                    label="Transcript Length",
                    choices=["short", "medium", "long"],
                    value="medium",
                )
                create_transcript_button = gr.Button("Create Transcript")
                transcript_output = gr.Textbox(label="Generated Transcript", lines=20, interactive=False)
                transcript_status = gr.Textbox(label="Status", interactive=False)
                refresh_content_button.click(
                    fn=lambda: gr.Dropdown(choices=interface.update_available_files()["content"]),
                    inputs=[],
                    outputs=[content_file_dropdown],
                )
                create_transcript_button.click(
                    fn=interface.create_podcast_transcript,
                    inputs=[content_file_dropdown, detailed_transcript, transcript_length],
                    outputs=[transcript_output, transcript_status],
                )

            with gr.Tab("TTS Generator"):
                gr.Markdown("## Generate Podcast Audio")
                transcript_file_dropdown = gr.Dropdown(
                    label="Select Transcript File",
                    choices=interface.available_files["transcripts"],
                    interactive=True,
                )
                refresh_transcript_button = gr.Button("Refresh Files")
                model_type = gr.Radio(
                    label="TTS Model",
                    choices=["fish", "parler", "bark", "coqui"],
                    value="fish",
                    info="Fish Speech is the default cloud path. Bark, Parler, and Coqui are local backends.",
                )
                speaker1_voice = gr.Dropdown(
                    label="Speaker 1 Voice (Parler)",
                    choices=["male_clear", "male_expressive", "male_deep", "male_casual"],
                    value="male_clear",
                )
                speaker2_voice = gr.Dropdown(
                    label="Speaker 2 Voice (Parler)",
                    choices=["female_expressive", "female_clear", "female_warm", "female_casual"],
                    value="female_expressive",
                )
                speaker3_voice = gr.Dropdown(
                    label="Speaker 3 Voice (Parler)",
                    choices=[
                        "male_expressive",
                        "male_clear",
                        "male_deep",
                        "male_casual",
                        "female_expressive",
                        "female_clear",
                        "female_warm",
                        "female_casual",
                    ],
                    value="male_expressive",
                )
                fish_reference = gr.Textbox(
                    label="Fish Voice Reference ID",
                    placeholder="Optional cloud voice ID",
                    value=os.environ.get("FISH_REFERENCE_ID", ""),
                )
                fish_emotion = gr.Dropdown(
                    label="Emotion Style",
                    choices=["(neutral)", "[excited]", "[whisper]", "[professional broadcast tone]", "[calm]", "[cheerful]", "[serious]"],
                    value="(neutral)",
                )
                generate_audio_button = gr.Button("Generate Audio")
                audio_output = gr.Audio(label="Generated Audio", interactive=False)
                audio_status = gr.Textbox(label="Status", interactive=False)
                refresh_transcript_button.click(
                    fn=lambda: gr.Dropdown(choices=interface.update_available_files()["transcripts"]),
                    inputs=[],
                    outputs=[transcript_file_dropdown],
                )
                generate_audio_button.click(
                    fn=interface.generate_podcast_audio,
                    inputs=[
                        transcript_file_dropdown,
                        model_type,
                        speaker1_voice,
                        speaker2_voice,
                        speaker3_voice,
                        fish_reference,
                        fish_emotion,
                    ],
                    outputs=[audio_output, audio_status],
                )

        gr.Markdown("---\n*planeLLM: Bite-sized podcasts to learn about anything powered by the OCI GenAI Service*")

    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(share=os.environ.get("PLANELLM_GRADIO_SHARE", "0") == "1")
