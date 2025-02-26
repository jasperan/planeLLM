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
"""

import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

from typing import Dict, Any, Optional
import oci
import yaml
import json
import pickle
import time
import os
import argparse

class PodcastWriter:
    """Class for transforming educational content into podcast transcript format."""
    
    def __init__(self, config_file: str = 'config.yaml', speakers: int = 2) -> None:
        """Initialize PodcastWriter with configuration.
        
        Args:
            config_file: Path to YAML configuration file
            speakers: Number of speakers in the conversation (2 or 3)
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If speakers is not 2 or 3
        """
        if speakers not in [2, 3]:
            raise ValueError("Number of speakers must be 2 or 3")
        
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
        self.speakers = speakers
        
        # Set up system prompt based on number of speakers
        if speakers == 2:
            self.system_prompt = """You are an expert podcast writer. Transform the following educational content into an engaging, conversational podcast transcript between two speakers:
            
            Speaker 1: An expert educator who explains concepts clearly and engagingly
            Speaker 2: A curious student who asks thoughtful questions and occasionally summarizes key points
            
            Guidelines:
            1. Start with a brief introduction by Speaker 1 welcoming listeners and introducing the topic
            2. Organize the content into a logical flow of conversation
            3. Include thoughtful questions from Speaker 2 that help transition between subtopics
            4. Ensure Speaker 1 explains concepts in an accessible, conversational way
            5. Include occasional moments where Speaker 2 summarizes or reflects on what they've learned
            6. End with a brief conclusion summarizing key takeaways
            7. Keep the tone educational but conversational and engaging
            8. Format the transcript with clear speaker labels (Speaker 1: and Speaker 2:)
            9. Aim for approximately 10-15 exchanges between speakers
            
            Here's the educational content to transform:
            
            """
        else:  # 3 speakers
            self.system_prompt = """You are an expert podcast writer. Transform the following educational content into an engaging, conversational podcast transcript between three speakers:
            
            Speaker 1: The primary expert who explains core concepts clearly and engagingly
            Speaker 2: A curious student who asks thoughtful questions
            Speaker 3: A secondary expert who adds additional context, examples, and occasionally friendly debate
            
            Guidelines:
            1. Start with a brief introduction by Speaker 1 welcoming listeners and introducing the topic
            2. Organize the content into a logical flow of conversation
            3. Include thoughtful questions from Speaker 2 that help transition between subtopics
            4. Ensure Speakers 1 and 3 explain concepts in an accessible, conversational way
            5. Include occasional friendly disagreement or different perspectives between Speakers 1 and 3
            6. End with a brief conclusion where all three speakers share final thoughts
            7. Keep the tone educational but conversational and engaging
            8. Format the transcript with clear speaker labels (Speaker 1:, Speaker 2:, and Speaker 3:)
            9. Aim for approximately 15-20 total exchanges between speakers
            
            Here's the educational content to transform:
            
            """
        
        # Initialize execution time tracking
        self.execution_times = {
            'start_time': 0,
            'total_time': 0,
            'llm_calls': []
        }

    def _call_llm(self, prompt: str) -> str:
        """Make a call to the OCI GenAI service.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response as a string
        """
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
        chat_request.max_tokens = 4000
        chat_request.temperature = 0.7
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
        
        # Track execution time
        duration = time.time() - start_time
        self.execution_times['llm_calls'].append({
            'prompt_length': len(prompt),
            'response_length': len(result),
            'duration': duration
        })
        
        return result

    def create_podcast_transcript(self, input_content: str) -> str:
        """Transform educational content into an engaging podcast transcript."""
        print("\nTransforming content into podcast format...")
        print(f"Input content length: {len(input_content)} characters")
        
        self.execution_times['start_time'] = time.time()
        
        print("Generating podcast script...")
        transcript = self._call_llm(self.system_prompt + input_content)
        
        # Calculate total execution time
        self.execution_times['total_time'] = time.time() - self.execution_times['start_time']
        
        print(f"\nTranscript generated in {self.execution_times['total_time']:.2f} seconds")
        print(f"Transcript length: {len(transcript)} characters")
        
        # Generate timestamp for file naming
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        transcript_file = f"podcast_transcript_{timestamp}.txt"
        
        # Save as text for easy reading
        print(f"Saving transcript to file: {transcript_file}...")
        os.makedirs('./resources', exist_ok=True)
        with open(f'./resources/{transcript_file}', 'w', encoding='utf-8') as file:
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
        summary.append(f"  - LLM Processing Time: {llm_call['duration']:.2f} seconds")
        summary.append(f"Total Execution Time: {self.execution_times['total_time']:.2f} seconds")
        
        return "\n".join(summary) 