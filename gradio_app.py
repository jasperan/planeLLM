#!/usr/bin/env python
"""
Launcher script for the planeLLM Gradio interface.

This script provides a simple way to launch the Gradio interface
without having to import the module directly.

Usage:
    python gradio_app.py
"""

# Import directly from the modules in the root directory
import os
import gradio as gr
import time
from typing import Dict, List, Tuple, Any, Optional
import json

# Import planeLLM components
from topic_explorer import TopicExplorer
from lesson_writer import PodcastWriter
from tts_generator import TTSGenerator

# Create resources directory if it doesn't exist
os.makedirs('./resources', exist_ok=True)

class PlaneLLMInterface:
    """Main class for the Gradio interface of planeLLM."""
    
    def __init__(self):
        """Initialize the interface components."""
        # Initialize components
        self.topic_explorer = TopicExplorer()
        self.podcast_writer = PodcastWriter()
        
        # We'll initialize the TTS generator only when needed to save memory
        self.tts_generator = None
        
        # Track available files
        self.update_available_files()
    
    def update_available_files(self) -> Dict[str, List[str]]:
        """Update and return lists of available files by type."""
        resources_dir = './resources'
        
        # Ensure directory exists
        os.makedirs(resources_dir, exist_ok=True)
        
        # Get all files in resources directory
        all_files = os.listdir(resources_dir)
        
        # Filter by type
        self.available_files = {
            'content': [f for f in all_files if f.endswith('.txt') and ('content' in f or 'raw_lesson' in f)],
            'questions': [f for f in all_files if f.endswith('.txt') and 'questions' in f],
            'transcripts': [f for f in all_files if f.endswith('.txt') and 'podcast' in f],
            'audio': [f for f in all_files if f.endswith('.mp3')]
        }
        
        return self.available_files
    
    def generate_topic_content(self, topic: str, progress=gr.Progress()) -> Tuple[str, str, str]:
        """Generate educational content about a topic.
        
        Args:
            topic: The topic to explore
            progress: Gradio progress indicator
            
        Returns:
            Tuple of (questions, content, status message)
        """
        if not topic:
            return "", "", "Error: Please enter a topic"
        
        try:
            progress(0, desc="Initializing...")
            
            # Generate timestamp for file naming
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            questions_file = f"questions_{timestamp}.txt"
            content_file = f"content_{timestamp}.txt"
            
            progress(0.1, desc="Generating questions...")
            questions = self.topic_explorer.generate_questions(topic)
            
            # Save questions to file
            with open(f"./resources/{questions_file}", 'w', encoding='utf-8') as f:
                questions_text = f"# Questions for {topic}\n\n"
                for i, q in enumerate(questions, 1):
                    questions_text += f"{i}. {q}\n"
                f.write(questions_text)
            
            progress(0.3, desc="Exploring questions...")
            # Generate content for each question
            results = {}
            for i, question in enumerate(questions):
                progress(0.3 + (0.6 * (i / len(questions))), 
                         desc=f"Exploring question {i+1}/{len(questions)}")
                response = self.topic_explorer.explore_question(question)
                results[question] = response
            
            # Combine content
            full_content = f"# {topic}\n\n"
            for question, response in results.items():
                full_content += f"# {question}\n\n{response}\n\n"
            
            # Save content to file
            with open(f"./resources/{content_file}", 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            progress(1.0, desc="Done!")
            self.update_available_files()
            
            return questions_text, full_content, f"Content generated successfully and saved to {content_file}"
            
        except Exception as e:
            return "", "", f"Error: {str(e)}"
    
    def create_podcast_transcript(self, content_file: str, detailed_transcript: bool, progress=gr.Progress()) -> Tuple[str, str]:
        """Create podcast transcript from content file.
        
        Args:
            content_file: Name of content file to use
            detailed_transcript: Whether to process each question individually
            progress: Gradio progress indicator
            
        Returns:
            Tuple of (transcript, status message)
        """
        if not content_file:
            return "", "Error: Please select a content file"
        
        try:
            progress(0, desc="Reading content file...")
            
            # Generate timestamp for file naming
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Read content from file
            with open(f"./resources/{content_file}", 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Initialize podcast writer
            self.podcast_writer = PodcastWriter()
            
            if detailed_transcript:
                progress(0.2, desc="Generating detailed podcast transcript (processing each question individually)...")
                transcript = self.podcast_writer.create_detailed_podcast_transcript(content)
                transcript_type = "detailed"
            else:
                progress(0.2, desc="Generating standard podcast transcript...")
                transcript = self.podcast_writer.create_podcast_transcript(content)
                transcript_type = "standard"
            
            # Transcript is saved by the PodcastWriter class
            # Find the most recently created transcript file
            transcript_files = [f for f in os.listdir('./resources') 
                              if f.startswith('podcast_transcript_') and f.endswith(f'{timestamp}.txt')]
            
            if transcript_files:
                transcript_file = transcript_files[0]
            else:
                # Fallback - save transcript to file
                transcript_file = f"podcast_transcript_{transcript_type}_{timestamp}.txt"
                with open(f"./resources/{transcript_file}", 'w', encoding='utf-8') as f:
                    f.write(transcript)
            
            progress(1.0, desc="Done!")
            self.update_available_files()
            
            return transcript, f"Transcript generated successfully and saved to {transcript_file}"
            
        except Exception as e:
            return "", f"Error: {str(e)}"
    
    def generate_podcast_audio(self, transcript_file: str, model_type: str, 
                              speaker1_voice: str = "male_clear", 
                              speaker2_voice: str = "female_expressive", 
                              speaker3_voice: str = "male_expressive", 
                              progress=gr.Progress()) -> Tuple[str, str]:
        """Generate podcast audio from transcript.
        
        Args:
            transcript_file: Name of transcript file to use
            model_type: TTS model to use ('parler', 'bark', or 'coqui')
            speaker1_voice: Voice style for Speaker 1
            speaker2_voice: Voice style for Speaker 2
            speaker3_voice: Voice style for Speaker 3
            progress: Gradio progress indicator
            
        Returns:
            Tuple of (audio path, status message)
        """
        if not transcript_file:
            return "", "Error: Please select a transcript file"
            
        # Validate model type - only allow parler and bark for now
        if model_type == "coqui":
            return "", f"Error: The {model_type} model is temporarily disabled. Please use the Parler or Bark model instead."
        
        try:
            progress(0, desc=f"Initializing {model_type} model...")
            
            # Initialize TTS generator if needed
            try:
                if self.tts_generator is None or self.tts_generator.model_type != model_type:
                    self.tts_generator = TTSGenerator(model_type=model_type)
                    
                    # Check if Parler was requested but fell back to Bark
                    if model_type == "parler" and not getattr(self.tts_generator, "parler_available", False):
                        progress(0.05, desc="Parler TTS not available, using Bark as fallback...")
                        return "", "Error: Parler TTS is not available. Please install it with: pip install git+https://github.com/huggingface/parler-tts.git"
                    
                    # Check if Coqui was requested but fell back to Bark
                    if model_type == "coqui" and not getattr(self.tts_generator, "coqui_available", False):
                        progress(0.05, desc="Coqui TTS not available, using Bark as fallback...")
                        return "", "Error: Coqui TTS is not available. Please install it with: pip install TTS"
            except ImportError as e:
                if "parler" in str(e).lower():
                    return "", "Error: Parler TTS module is not installed. Please run: pip install git+https://github.com/huggingface/parler-tts.git"
                elif "tts" in str(e).lower():
                    return "", "Error: Coqui TTS module is not installed. Please run: pip install TTS"
                else:
                    raise
            
            # Check for FFmpeg
            if not getattr(self.tts_generator, "ffmpeg_available", True):
                return "", "Error: FFmpeg/ffprobe not found. Please install FFmpeg to generate audio. Ubuntu/Debian: sudo apt-get install ffmpeg"
            
            # Set voice types for Parler TTS
            if model_type == "parler" and hasattr(self.tts_generator, "set_voice_type"):
                progress(0.05, desc="Setting voice types...")
                self.tts_generator.set_voice_type("Speaker 1", speaker1_voice)
                self.tts_generator.set_voice_type("Speaker 2", speaker2_voice)
                self.tts_generator.set_voice_type("Speaker 3", speaker3_voice)
            
            # Generate timestamp for file naming
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            audio_file = f"podcast_{timestamp}.mp3"
            audio_path = f"./resources/{audio_file}"
            
            progress(0.1, desc="Generating podcast audio...")
            
            # Read transcript from file
            transcript_path = f"./resources/{transcript_file}"
            if os.path.exists(transcript_path) and os.path.isfile(transcript_path):
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read()
                
                # Generate podcast audio - pass transcript text directly instead of file path
                result_path = self.tts_generator.generate_podcast(transcript, output_path=audio_path)
                
                # Check if the result is an error file
                if isinstance(result_path, str) and result_path.endswith('.txt') and 'error' in result_path:
                    with open(result_path, 'r', encoding='utf-8') as f:
                        error_content = f.read()
                    return "", f"Error: {error_content.splitlines()[0]}"
                
                # Update the audio_path with the actual result path
                audio_path = result_path
            else:
                return "", f"Error: Transcript file not found at {transcript_path}"
            
            progress(1.0, desc="Done!")
            self.update_available_files()
            
            # Check if the audio file was actually created
            if isinstance(audio_path, str) and os.path.exists(audio_path) and os.path.isfile(audio_path) and audio_path.endswith('.mp3'):
                return audio_path, f"Podcast audio generated successfully and saved to {os.path.basename(audio_path)}"
            else:
                return "", f"Error: Failed to generate audio file. Please check the logs for details."
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error generating podcast audio: {error_details}")
            
            # Provide more helpful error messages
            if "Permission denied" in str(e):
                return "", "Error: Permission denied. Please check if the output file is open in another program."
            elif "No module named" in str(e):
                if "parler" in str(e).lower():
                    return "", "Error: Parler TTS module is not installed. Please run: pip install git+https://github.com/huggingface/parler-tts.git"
                elif "ffmpeg" in str(e).lower() or "ffprobe" in str(e).lower():
                    return "", "Error: FFmpeg/ffprobe not found. Please install FFmpeg to generate audio. Ubuntu/Debian: sudo apt-get install ffmpeg"
                else:
                    return "", f"Error: Missing module - {str(e)}"
            elif "ffprobe" in str(e) or "ffmpeg" in str(e):
                return "", "Error: FFmpeg/ffprobe not found. Please install FFmpeg to generate audio. Ubuntu/Debian: sudo apt-get install ffmpeg"
            elif "Is a directory" in str(e):
                return "", "Error: Invalid output path. Please check if the resources directory exists and is writable."
            else:
                return "", f"Error: {str(e)}"

def create_interface():
    """Create and launch the Gradio interface."""
    # Initialize the interface
    interface = PlaneLLMInterface()
    
    # Define the interface
    with gr.Blocks(title="planeLLM Interface") as app:
        gr.Markdown("# planeLLM: Educational Content Generation System")
        
        # Create tabs for different components
        with gr.Tabs():
            # Topic Explorer Tab
            with gr.Tab("Topic Explorer"):
                gr.Markdown("## Generate Educational Content")
                
                with gr.Row():
                    topic_input = gr.Textbox(label="Topic", placeholder="Enter a topic (e.g., Ancient Rome, Quantum Physics)")
                    generate_button = gr.Button("Generate Content")
                
                with gr.Row():
                    with gr.Column():
                        questions_output = gr.Textbox(label="Generated Questions", lines=10, interactive=False)
                    with gr.Column():
                        content_output = gr.Textbox(label="Generated Content", lines=20, interactive=False)
                
                status_output = gr.Textbox(label="Status", interactive=False)
                
                # Connect the button to the function
                generate_button.click(
                    fn=interface.generate_topic_content,
                    inputs=[topic_input],
                    outputs=[questions_output, content_output, status_output]
                )
            
            # Lesson Writer Tab
            with gr.Tab("Lesson Writer"):
                gr.Markdown("## Create Podcast Transcript")
                
                with gr.Row():
                    # Dropdown for selecting content file
                    content_file_dropdown = gr.Dropdown(
                        label="Select Content File",
                        choices=interface.available_files['content'],
                        interactive=True
                    )
                    refresh_content_button = gr.Button("Refresh Files")
                
                with gr.Row():
                    detailed_transcript = gr.Checkbox(
                        label="Detailed Processing",
                        value=True,
                        info="Process each question individually for more detailed content (recommended)"
                    )
                
                create_transcript_button = gr.Button("Create Transcript")
                
                transcript_output = gr.Textbox(label="Generated Transcript", lines=20, interactive=False)
                transcript_status = gr.Textbox(label="Status", interactive=False)
                
                # Connect buttons to functions
                refresh_content_button.click(
                    fn=lambda: gr.Dropdown(choices=interface.update_available_files()['content']),
                    inputs=[],
                    outputs=[content_file_dropdown]
                )
                
                create_transcript_button.click(
                    fn=interface.create_podcast_transcript,
                    inputs=[content_file_dropdown, detailed_transcript],
                    outputs=[transcript_output, transcript_status]
                )
            
            # TTS Generator Tab
            with gr.Tab("TTS Generator"):
                gr.Markdown("## Generate Podcast Audio")
                
                with gr.Row():
                    # Dropdown for selecting transcript file
                    transcript_file_dropdown = gr.Dropdown(
                        label="Select Transcript File",
                        choices=interface.available_files['transcripts'],
                        interactive=True
                    )
                    refresh_transcript_button = gr.Button("Refresh Files")
                
                with gr.Row():
                    model_type = gr.Radio(
                        label="TTS Model",
                        choices=["parler", "bark", "coqui"],
                        value="parler",
                        info="Parler: Fast with good quality, Bark: High quality but slow, Coqui: High quality with natural intonation (currently disabled)"
                    )
                    
                    # Add a note about disabled models
                    gr.Markdown("*Note: Currently Parler and Bark are fully supported. Coqui option will be enabled in a future update.*")
                
                # Add voice selection options for Parler
                with gr.Row(visible=True) as parler_options:
                    with gr.Column():
                        speaker1_voice = gr.Dropdown(
                            label="Speaker 1 Voice (Expert)",
                            choices=["male_clear", "male_expressive", "male_deep", "male_casual"],
                            value="male_clear",
                            info="Select voice style for Speaker 1 (expert)"
                        )
                    
                    with gr.Column():
                        speaker2_voice = gr.Dropdown(
                            label="Speaker 2 Voice (Student)",
                            choices=["female_expressive", "female_clear", "female_warm", "female_casual"],
                            value="female_expressive",
                            info="Select voice style for Speaker 2 (student)"
                        )
                    
                    with gr.Column():
                        speaker3_voice = gr.Dropdown(
                            label="Speaker 3 Voice (Second Expert)",
                            choices=["male_expressive", "male_clear", "male_deep", "male_casual", 
                                    "female_expressive", "female_clear", "female_warm", "female_casual"],
                            value="male_expressive",
                            info="Select voice style for Speaker 3 (second expert)"
                        )
                
                # Show/hide voice options based on model selection
                def update_voice_options(model):
                    return gr.update(visible=(model == "parler"))
                
                model_type.change(
                    fn=update_voice_options,
                    inputs=[model_type],
                    outputs=[parler_options]
                )
                
                generate_audio_button = gr.Button("Generate Audio")
                
                with gr.Row():
                    audio_output = gr.Audio(label="Generated Audio", interactive=False)
                
                audio_status = gr.Textbox(label="Status", interactive=False)
                
                # Connect buttons to functions
                refresh_transcript_button.click(
                    fn=lambda: gr.Dropdown(choices=interface.update_available_files()['transcripts']),
                    inputs=[],
                    outputs=[transcript_file_dropdown]
                )
                
                generate_audio_button.click(
                    fn=interface.generate_podcast_audio,
                    inputs=[transcript_file_dropdown, model_type, 
                           speaker1_voice, speaker2_voice, speaker3_voice],
                    outputs=[audio_output, audio_status]
                )
        
        # Add a footer
        gr.Markdown("---\n*planeLLM: Bite-sized podcasts to learn about anything powered by the OCI GenAI Service*")
    
    # Launch the interface
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True) 