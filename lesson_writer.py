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
    
    # For question-by-question processing:
    writer = PodcastWriter()
    transcript = writer.create_detailed_podcast_transcript(content)
"""

import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

from typing import Dict, Any, Optional, List
import oci
import yaml
import json
import pickle
import time
import os
import argparse
import re

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
        
        # Fixed token limit to avoid exceeding the 4000 token limit
        self.max_tokens = 4000
        self.exchange_count_2speakers = "10-15"
        self.exchange_count_3speakers = "15-20"
        
        # Set up system prompt based on number of speakers
        if speakers == 2:
            self.system_prompt = f"""You are an expert podcast writer. Transform the following educational content into an engaging, conversational podcast transcript between two speakers:
            
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
            12. Create a substantial, in-depth conversation that thoroughly explores the topic
            
            Here's the educational content to transform:
            
            """
        else:  # 3 speakers
            self.system_prompt = """You are an expert podcast writer who transforms educational content into engaging, conversational podcast transcripts. Your task is to create a natural, interactive dialogue between speakers that educates listeners on the given topic.

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
            12. Create a substantial, in-depth conversation that thoroughly explores the topic
            13. Make the conversation feel natural by including:
                - Occasional [laughs], [sighs], or [pauses] to show emotion
                - Interruptions where speakers build on each other's points
                - Informal language and conversational phrases ("you know", "I mean", etc.)
                - Reactions to what other speakers say ("That's fascinating!", "Wait, really?")
                - Brief personal anecdotes or experiences related to the topic
            14. Avoid making the dialogue too formal or lecture-like
            
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
        chat_request.max_tokens = self.max_tokens
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
        
    def extract_questions_from_content(self, content: str) -> List[str]:
        """Extract questions from the content.
        
        Args:
            content: The educational content
            
        Returns:
            List of questions extracted from the content
        """
        # Look for a questions section at the beginning
        questions = []
        
        # Try to find questions in a "Questions for X" section
        questions_section_match = re.search(r"# Questions for .+\n\n((?:\d+\.\s+.+\n?)+)", content)
        if questions_section_match:
            questions_text = questions_section_match.group(1)
            questions = re.findall(r"\d+\.\s+(.+)", questions_text)
            return questions
            
        # If no questions section, try to find headers or questions in the text
        headers = re.findall(r"#+\s+(.+)", content)
        if headers:
            # Filter out the main title
            if len(headers) > 1:
                return headers[1:]  # Skip the first header which is usually the title
            
        # If no headers, look for sentences ending with question marks
        question_sentences = re.findall(r"([^.!?]+\?)", content)
        if question_sentences:
            return question_sentences[:10]  # Limit to 10 questions
            
        # If all else fails, split the content into chunks
        paragraphs = content.split("\n\n")
        if len(paragraphs) >= 3:
            # Use the first few paragraphs as "questions"
            return paragraphs[1:min(6, len(paragraphs))]
            
        return ["Tell me about this topic"]  # Fallback
    
    def create_detailed_podcast_transcript(self, input_content: str) -> str:
        """Create a detailed podcast transcript by processing each question separately.
        
        This method extracts questions from the content and generates a separate
        conversation segment for each question, then combines them into a cohesive
        podcast transcript.
        
        Args:
            input_content: The educational content to transform
            
        Returns:
            The generated podcast transcript
        """
        print("\nCreating detailed podcast transcript by processing questions individually...")
        print(f"Input content length: {len(input_content)} characters")
        
        self.execution_times['start_time'] = time.time()
        
        # Extract questions or sections from the content
        questions = self.extract_questions_from_content(input_content)
        print(f"Extracted {len(questions)} questions/sections to process")
        
        # Generate introduction
        intro_prompt = """You are an expert podcast writer. Create ONLY the introduction section for a podcast about the following topic:

        {}
        
        Guidelines:
        1. Write only the introduction (first 2-3 exchanges)
        2. Include Speaker 1 welcoming listeners and introducing the topic
        3. Include Speaker 2 expressing interest and asking an initial question
        4. Format with Speaker 1: and Speaker 2: labels
        5. Keep it brief (about 150-200 words) and engaging
        6. Do NOT start exploring the topic in depth yet
        7. Make the conversation feel natural by including:
           - Occasional [laughs], [sighs], or [pauses] to show emotion
           - Informal language and conversational phrases
           - Genuine enthusiasm and curiosity
        """.format(input_content.split('\n\n')[0])
        
        print("Generating podcast introduction...")
        introduction = self._call_llm(intro_prompt)
        
        # Process each question
        segments = [introduction]
        for i, question in enumerate(questions):
            print(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            # Create a focused prompt for this question
            segment_prompt = """You are an expert podcast writer. Create a segment of a podcast conversation about the following question:

            {}
            
            Use the following educational content as reference:
            {}
            
            Guidelines:
            1. Write only the segment discussing this specific question/topic
            2. Start with Speaker 2 asking about this topic
            3. Have Speaker 1 provide a detailed, thorough explanation
            4. Include follow-up questions from Speaker 2 to explore the topic deeply
            5. Format with Speaker 1: and Speaker 2: labels
            6. Make the explanation detailed with examples, analogies, and real-world applications
            7. Aim for about 400-600 words for this segment
            8. Do NOT include an introduction or conclusion to the whole podcast
            9. Make the conversation feel natural by including:
               - Occasional [laughs], [sighs], or [pauses] to show emotion
               - Interruptions where speakers build on each other's points
               - Informal language and conversational phrases
               - Reactions to what other speakers say
            """.format(question, input_content)
            
            segment = self._call_llm(segment_prompt)
            segments.append(segment)
        
        # Generate conclusion
        conclusion_prompt = """You are an expert podcast writer. Create ONLY the conclusion section for a podcast about the following topic:

        {}
        
        The podcast has covered these main points:
        {}
        
        Guidelines:
        1. Write only the conclusion (last 2-3 exchanges)
        2. Include Speaker 1 summarizing key takeaways
        3. Include Speaker 2 reflecting on what they've learned
        4. Include a sign-off and thank you to listeners
        5. Format with Speaker 1: and Speaker 2: labels
        6. Keep it concise (about 150-200 words) and impactful
        7. Make the conclusion feel natural by including:
           - Expressions of gratitude between speakers
           - A touch of humor or emotion [laughs]
           - Informal closing remarks that sound conversational
        """.format(input_content.split('\n\n')[0], ', '.join(questions[:5]))
        
        print("Generating podcast conclusion...")
        conclusion = self._call_llm(conclusion_prompt)
        segments.append(conclusion)
        
        # Combine all segments
        full_transcript = "\n\n".join(segments)
        
        # Clean up any duplicate speaker labels or formatting issues
        full_transcript = re.sub(r"(Speaker \d+:)\s+\1", r"\1", full_transcript)
        
        # Calculate total execution time
        self.execution_times['total_time'] = time.time() - self.execution_times['start_time']
        
        print(f"\nDetailed transcript generated in {self.execution_times['total_time']:.2f} seconds")
        print(f"Transcript length: {len(full_transcript)} characters")
        
        # Generate timestamp for file naming
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        transcript_file = f"podcast_transcript_detailed_{timestamp}.txt"
        
        # Save as text for easy reading
        print(f"Saving transcript to file: {transcript_file}...")
        os.makedirs('./resources', exist_ok=True)
        with open(f'./resources/{transcript_file}', 'w', encoding='utf-8') as file:
            file.write(full_transcript)
        print("Transcript saved successfully!")
        
        return full_transcript

    def _generate_timing_summary(self) -> str:
        """Generate a summary of execution times."""
        summary = ["=== Execution Time Summary ==="]
        
        # Get LLM call stats
        total_calls = len(self.execution_times['llm_calls'])
        total_prompt_length = sum(call['prompt_length'] for call in self.execution_times['llm_calls'])
        total_response_length = sum(call['response_length'] for call in self.execution_times['llm_calls'])
        total_llm_time = sum(call['duration'] for call in self.execution_times['llm_calls'])
        
        summary.append("\nLLM Statistics:")
        summary.append(f"  - Total LLM Calls: {total_calls}")
        summary.append(f"  - Total Prompt Length: {total_prompt_length} characters")
        summary.append(f"  - Total Response Length: {total_response_length} characters")
        summary.append(f"  - Total LLM Processing Time: {total_llm_time:.2f} seconds")
        summary.append(f"Total Execution Time: {self.execution_times['total_time']:.2f} seconds")
        
        return "\n".join(summary) 