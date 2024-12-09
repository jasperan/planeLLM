"""Lesson Writer Module for planeLLM with three speakers."""

import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

from typing import Dict, Any, Optional
import oci
import yaml
import os
import time
from datetime import datetime

class PodcastWriter:
    """Class for converting educational content into podcast format using OCI GenAI service."""
    
    def __init__(self, config_file: str = 'config.yaml') -> None:
        """Initialize PodcastWriter with configuration."""
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

        # Define system prompt for podcast conversion with three speakers
        self.system_prompt = """You are a world-class podcast writer who has worked with top educational podcasters like Lex Fridman and Tim Ferriss.

Your task is to transform educational content into an engaging podcast conversation between three speakers:

Speaker 1: An expert historian who explains concepts clearly, uses great analogies, and shares relevant examples
Speaker 2: A curious student who asks insightful questions and occasionally goes on interesting tangents
Speaker 3: A developer who understands the technical aspects of building web services, connecting services with API's and has a working knowledge of devops as it pertains to AI workloads

The conversation should:
1. Feel natural with "umm", "hmm", and other verbal fillers
2. Include interruptions and clarifying questions from both Speaker 2 and Speaker 3
3. Use real-world examples and analogies
4. Have occasional tangents that make the content more engaging
5. Maintain an educational but conversational tone
6. Include technical perspectives and implementation details from Speaker 3
7. Create natural interactions between all three speakers

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
        chat_request.max_tokens = 4096
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
        json_result = vars(response)['data']
        
        # Store execution time
        self.execution_times['total_time'] = time.time() - start_time
        
        return json_result['chat_response']['choices'][0]['message']['content'][0]['text']

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