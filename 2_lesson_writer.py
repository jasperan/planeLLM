"""
Lesson Writer Module for planeLLM.

This module transforms educational content into an engaging podcast transcript format
using OCI's GenAI service. It takes the raw educational content and converts it into
a natural conversation between an expert and a student.

The module includes detailed execution time tracking and statistics for monitoring
LLM performance and content generation metrics.

Example:
    writer = PodcastWriter()
    transcript = writer.create_podcast_transcript(content)

Attributes:
    DEFAULT_CONFIG_FILE (str): Default path to configuration file

Classes:
    PodcastWriter: Main class for transcript generation
"""

import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

from typing import Dict, Any, Optional
import oci
import yaml
import json
import pickle
import os
import time
from datetime import datetime
import argparse

class PodcastWriter:
    """Class for converting educational content into podcast format using OCI GenAI service."""
    
    def __init__(self, config_file: str = 'config.yaml', speakers: int = 2) -> None:
        """Initialize PodcastWriter with configuration.
        
        Args:
            config_file: Path to YAML configuration file
            speakers: Number of speakers in the conversation
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
            oci.exceptions.ConfigFileNotFound: If OCI config is missing
        """
        with open(config_file, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        self.compartment_id: str = config_data['compartment_id']
        config = oci.config.from_file('~/.oci/config', config_data['config_profile'])
        
        self.genai_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config,
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240)
        )
        
        self.model_id: str = config_data['model_id']
        self.execution_times: Dict[str, float] = {
            'total_time': 0
        }

        # Define system prompt for podcast conversion
        self.system_prompt = """You are a world-class podcast writer who has worked with top educational podcasters like Lex Fridman and Tim Ferriss.

Your task is to transform educational content into an engaging podcast conversation between two speakers:

Speaker 1: An expert historian who explains concepts clearly, uses great analogies, and shares relevant examples
Speaker 2: A curious student who asks insightful questions and occasionally goes on interesting tangents

The conversation should:
1. Feel natural with "umm", "hmm", and other verbal fillers
2. Include interruptions and clarifying questions
3. Use real-world examples and analogies
4. Have occasional tangents that make the content more engaging
5. Maintain an educational but conversational tone

Start directly with Speaker 1's introduction. Do not include episode titles or chapters.

Here's the content to transform:

"""

    def _call_llm(self, prompt: str) -> str:
        """Make a call to the OCI GenAI service."""
        start_time = time.time()
        
        # Create message content
        content = oci.generative_ai_inference.models.TextContent()
        content.text = prompt

        # Create message
        message = oci.generative_ai_inference.models.Message()
        message.role = "USER"
        message.content = [content]

        # Create chat request
        chat_request = oci.generative_ai_inference.models.GenericChatRequest()
        chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
        chat_request.messages = [message]
        chat_request.max_tokens = 50000
        chat_request.temperature = 1.0  # Higher temperature for more creative podcast-style content
        chat_request.frequency_penalty = 0.0
        chat_request.presence_penalty = 0
        chat_request.top_p = 0.7
        chat_request.top_k = -1

        # Create chat details
        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=self.model_id
        )
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = self.compartment_id

        # Make the API call
        response = self.genai_client.chat(chat_detail)
        json_result = json.loads(str(vars(response)['data']))
        result = json_result['chat_response']['choices'][0]['message']['content'][0]['text']
        
        # Store execution time
        execution_time = time.time() - start_time
        self.execution_times['llm_calls'].append({
            'prompt_length': len(prompt),
            'response_length': len(result),
            'execution_time': execution_time
        })
        
        return result

    def create_podcast_transcript(self, input_content: str) -> str:
        """Transform educational content into an engaging podcast transcript."""
        print("\nTransforming content into podcast format...")
        print(f"Input content length: {len(input_content)} characters")
        
        start_time = time.time()
        
        print("Generating podcast script...")
        transcript = self._call_llm(self.system_prompt + input_content)
        
        duration = time.time() - start_time
        print(f"\nTranscript generated in {duration:.2f} seconds")
        print(f"Transcript length: {len(transcript)} characters")
        
        # Save as text for easy reading
        print("Saving transcript to file...")
        with open('./resources/podcast_transcript.txt', 'w', encoding='utf-8') as file:
            file.write(transcript)
        print("Transcript saved successfully!")
        
        return transcript

    def _generate_timing_summary(self) -> str:
        """Generate a summary of execution times."""
        summary = ["=== Execution Time Summary ==="]
        
        # Get the single LLM call stats
        llm_call = self.execution_times['llm_calls'][0]
        
        summary.append("\nLLM Statistics:")
        summary.append(f"  - Prompt Length: {llm_call['prompt_length']} characters")
        summary.append(f"  - Response Length: {llm_call['response_length']} characters")
        summary.append(f"  - LLM Processing Time: {llm_call['execution_time']:.2f} seconds")
        summary.append(f"Total Execution Time: {self.execution_times['total_time']:.2f} seconds")
        
        return "\n".join(summary)

def main():
    """Run the podcast writer with command line arguments."""
    parser = argparse.ArgumentParser(description='Convert educational content to podcast format')
    parser.add_argument('--speakers', type=int, choices=[2, 3], default=2,
                      help='Number of speakers in the conversation (default: 2)')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file (default: config.yaml)')
    
    args = parser.parse_args()
    
    print(f"\nInitializing PodcastWriter with {args.speakers} speakers...")
    writer = PodcastWriter(config_file=args.config, speakers=args.speakers)
    
    print("Reading content from raw_lesson_content.txt...")
    try:
        with open('./resources/raw_lesson_content.txt', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("Error: raw_lesson_content.txt not found in resources directory.")
        print("Please run step 1 (topic_explorer.py) first.")
        return
    
    writer.create_podcast_transcript(content)

if __name__ == '__main__':
    main() 