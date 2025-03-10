"""
Topic Explorer Module for planeLLM.

This module is responsible for generating educational content about a given topic using
OCI's GenAI service with Llama 3.1 70B. It follows a two-step process:
1. Generate relevant questions about the topic
2. Create detailed answers for each question

The module uses a class-based approach with comprehensive error handling and
execution time tracking. It saves both the questions and content separately.
It utilizes multithreading to significantly improve performance.

Example:
    explorer = TopicExplorer()
    content = explorer.generate_full_content("Ancient Rome")
"""

import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

import oci
import yaml
import json
from typing import List, Dict, Any, Optional, Tuple
import time
import os
import argparse
import tiktoken
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Lock, Event

class RateLimiter:
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.interval = 60  # 1 minute in seconds
        self.requests = []
        self.lock = Lock()
        
    def acquire(self):
        with self.lock:
            now = time.time()
            # Remove requests older than the interval
            self.requests = [req_time for req_time in self.requests if now - req_time <= self.interval]
            
            # if we hit the limit, wait
            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.interval - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.requests = self.requests[1:]  # Remove oldest request
            
            # Add current request
            self.requests.append(now)

class TopicExplorer:
    """Class for generating educational content about a topic using OCI GenAI service."""
    
    def __init__(self, config_file: str = 'config.yaml', max_workers: int = 10) -> None:
        """Initialize TopicExplorer with configuration.
        
        Args:
            config_file: Path to YAML configuration file
            max_workers: Maximum number of worker threads to use for parallel processing
        
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
        self.execution_times: Dict[str, Any] = {
            'questions_generation': 0,
            'responses': {}
        }

        # Configure multithreading
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(max_requests_per_minute=60)
        self.response_lock = Lock()

    def _generate_question_batch(self, topic: str, batch_id: int) -> List[str]:
        """Generate a batch of questions about the topic."""
        prompt = f"""You are an expert researcher and educator. Generate 3-4 specific, detailed questions about {topic}.
        The questions should:
        1. Cover different aspects of the topic (history, key concepts, applications, impact, etc.)
        2. Include both well-known fundamentals and lesser-known interesting facts
        3. Be specific enough to generate detailed, educational responses
        4. Avoid overly broad questions that can't be answered thoroughly
        
        For batch {batch_id}, focus on {'foundational concepts' if batch_id == 1 else 'advanced concepts' if batch_id == 2 else 'real-world applications and impact'}.
        
        Format: Return only the questions, one per line, without numbering."""
        
        response = self._call_llm(prompt)
        questions = [q.strip() for q in response.split('\n') if q.strip() and '?' in q]
        return questions

    def generate_questions(self, topic: str) -> List[str]:
        """Generate relevant questions about the topic using parallel processing."""
        print(f"\nGenerating questions about '{topic}'...")
        start_time = time.time()
        
        all_questions = []
        
        # Use ThreadPoolExecutor to generate questions in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks for different batches of questions
            futures = [
                executor.submit(self._generate_question_batch, topic, batch_id)
                for batch_id in range(1, 4)  # 3 batches with different focus
            ]
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    batch_questions = future.result()
                    all_questions.extend(batch_questions)
                except Exception as e:
                    print(f"Error generating questions: {str(e)}")
        
        # Ensure we have a reasonable number of questions
        if len(all_questions) > 10:
            all_questions = all_questions[:10]
        
        duration = time.time() - start_time
        self.execution_times['questions_generation'] = duration
        print(f"Generated {len(all_questions)} questions in {duration:.2f} seconds")
        
        return all_questions

    def _call_llm(self, prompt: str) -> str:
        """Make a rate-limited call to the OCI GenAI service."""
        self.rate_limiter.acquire()
        return self._make_llm_call(prompt)

    def _make_llm_call(self, prompt: str) -> str:
        """Internal method to make the actual LLM call."""
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
        chat_request.max_tokens = 3850 # max is 128K (https://docs.oracle.com/en-us/iaas/Content/generative-ai/chat-models.htm)
        chat_request.temperature = 0.5 # 0.7?
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
        try:
            response = self.genai_client.chat(chat_detail)
            json_result = json.loads(str(vars(response)['data']))
            return json_result['chat_response']['choices'][0]['message']['content'][0]['text']
        except Exception as e:
            print(f"Error calling LLM: {str(e)}")
            # Retry once after a short delay
            time.sleep(2)
            try:
                response = self.genai_client.chat(chat_detail)
                json_result = json.loads(str(vars(response)['data']))
                return json_result['chat_response']['choices'][0]['message']['content'][0]['text']
            except Exception as e:
                print(f"Error on retry: {str(e)}")
                return f"Error generating content: {str(e)}"

    def _explore_question_thread(self, question: str, results: Dict[str, str]):
        """Thread-safe question exploration."""
        start_time = time.time()
        response = self._call_llm(f"""As an expert educator, provide a detailed, engaging response to this question:
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
        """)
        
        with self.response_lock:
            results[question] = response
            self.execution_times['responses'][question] = time.time() - start_time

    def generate_full_content(self, topic: str) -> str:
        """Generate complete content about the topic using multithreading."""
        total_start_time = time.time()
        
        print(f"Generating questions about {topic}...")
        questions = self.generate_questions(topic)
        print(f'Generated {len(questions)} questions')
        
        # Initialize content without questions
        full_content = f"# {topic}\n\n"
        # Prepare questions content separately
        questions_content = f"# Questions for {topic}\n\n"
        
        # Use a dictionary to store results in order
        results = {}
        
        print(f"Exploring {len(questions)} questions in parallel...")
        # Create and start threads for each question
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for question in questions:
                futures.append(
                    executor.submit(self._explore_question_thread, question, results)
                )
            
            # Wait for all threads to complete
            for i, future in enumerate(as_completed(futures)):
                print(f"Completed {i+1}/{len(futures)} questions")
                future.result()
        
        # Process results in order
        for i, question in enumerate(questions, 1):
            response = results[question]
            full_content += f"## {question}\n\n{response}\n\n"
            questions_content += f"{i}. {question}\n"
            print(f'Question {i}: Generated {len(response.split())} words')
            print(f'Question execution time: {self.execution_times["responses"][question]:.2f} seconds')
        
        # Calculate and display timing summary
        total_time = time.time() - total_start_time
        timing_summary = self._generate_timing_summary(total_time)
        print("\n" + timing_summary)
        
        # Create resources directory if it doesn't exist
        os.makedirs('./resources', exist_ok=True)
        
        # Generate timestamp for file naming
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        content_file = f"raw_lesson_content_{timestamp}.txt"
        questions_file = f"questions_{timestamp}.txt"
        
        # Save the content and questions separately, without timing summary
        with open(f'./resources/{content_file}', 'w', encoding='utf-8') as f:
            f.write(full_content)
            
        with open(f'./resources/{questions_file}', 'w', encoding='utf-8') as f:
            f.write(questions_content)
        
        print(f"Content saved to ./resources/{content_file}")
        print(f"Questions saved to ./resources/{questions_file}")
        
        return full_content

    def _generate_timing_summary(self, total_time: float) -> str:
        """Generate a summary of execution times."""
        summary = ["=== Execution Time Summary ==="]
        summary.append(f"Questions Generation: {self.execution_times['questions_generation']:.2f} seconds")
        
        # Individual question times
        summary.append("\nIndividual Question Times:")
        for question, time_taken in self.execution_times['responses'].items():
            summary.append(f"- {question[:50]}...: {time_taken:.2f} seconds")
        
        # Calculate average response time
        avg_response_time = sum(self.execution_times['responses'].values()) / len(self.execution_times['responses'])
        summary.append(f"\nAverage Response Time: {avg_response_time:.2f} seconds")
        summary.append(f"Total Execution Time: {total_time:.2f} seconds")
        
        return "\n".join(summary)

    def explore_question(self, question: str) -> str:
        """Generate detailed content for a specific question."""
        print(f"\nExploring: {question}")
        start_time = time.time()
        
        prompt = f"""As an expert educator, provide a detailed, engaging response to this question:
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
        print("Generating detailed response...")
        response = self._call_llm(prompt)
        
        duration = time.time() - start_time
        tokens = len(response.split())
        print(f"Generated {tokens} words in {duration:.2f} seconds")
        
        return response

def count_tokens(text: str) -> int:
    """Count the number of tokens in the text using GPT tokenizer."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text)) 