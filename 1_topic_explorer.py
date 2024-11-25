import oci
import yaml
import json
from typing import List, Dict
import time
import os
import argparse
import tiktoken

class TopicExplorer:
    def __init__(self, config_file: str = 'config.yaml'):
        # Load configuration
        with open(config_file, 'r') as file:
            config_data = yaml.safe_load(file)
        
        self.compartment_id = config_data['compartment_id']
        config = oci.config.from_file('~/.oci/config', config_data['config_profile'])
        
        # Initialize OCI client
        self.genai_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config,
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240)
        )
        
        # Load model ID from config
        self.model_id = config_data['model_id']

        # Add timing dictionary to store execution times
        self.execution_times = {
            'questions_generation': 0,
            'responses': {}
        }

    def generate_questions(self, topic: str) -> List[str]:
        """Generate relevant questions about the topic."""
        start_time = time.time()
        
        prompt = f"""You are an expert researcher. Generate 8-10 specific, detailed questions about {topic}.
        The questions should:
        1. Provide an introduction to the topic at hand
        2. Cover different aspects and time periods
        3. Include both well-known and lesser-known facts
        3. Focus on simplicity and allowing people to completely learn about the topic by answering these questions
        4. Be specific enough to generate detailed responses
        
        Format: Return only the questions, one per line."""

        response = self._call_llm(prompt)
        questions = [q.strip() for q in response.split('\n') if q.strip() and '?' in q]
        
        # Store execution time
        self.execution_times['questions_generation'] = time.time() - start_time
        return questions

    def explore_question(self, question: str) -> str:
        """Generate detailed content for a specific question."""
        start_time = time.time()
        
        prompt = f"""As an expert researcher, provide a detailed response to this question:
        {question}
        
        Your response should:
        1. Be detailed and engaging (aim for 500-700 words)
        2. Include specific dates, names, and places
        3. Share interesting anecdotes or lesser-known facts
        4. Connect events to their broader context
        """
        
        response = self._call_llm(prompt)
        
        # Store execution time
        self.execution_times['responses'][question] = time.time() - start_time
        return response

    def _call_llm(self, prompt: str) -> str:
        """Make a call to the OCI GenAI service."""
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
        chat_request.max_tokens = 9192 # max is 128K (https://docs.oracle.com/en-us/iaas/Content/generative-ai/chat-models.htm)
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
        response = self.genai_client.chat(chat_detail)
        json_result = json.loads(str(vars(response)['data']))
        return json_result['chat_response']['choices'][0]['message']['content'][0]['text']

    def generate_full_content(self, topic: str) -> str:
        """Generate complete content about the topic through multiple questions."""
        total_start_time = time.time()
        
        print(f"Generating questions about {topic}...")
        questions = self.generate_questions(topic)

        print('Questions: {}'.format(questions))
        
        # Initialize content without questions
        full_content = f"# {topic}\n\n"
        # Prepare questions content separately
        questions_content = f"# Questions for {topic}\n\n"
        
        for i, question in enumerate(questions, 1):
            print(f"Exploring question {i}/{len(questions)}: {question}")
            response = self.explore_question(question)
            # Add only the response to full_content
            full_content += f"{response}\n\n"
            # Add numbered question to questions content
            questions_content += f"{i}. {question}\n"
            
            print('Generated {} tokens'.format(len(response)))
            print(f'Question execution time: {self.execution_times["responses"][question]:.2f} seconds')
        
        # Calculate and display timing summary
        total_time = time.time() - total_start_time
        timing_summary = self._generate_timing_summary(total_time)
        print("\n" + timing_summary)
        
        # Create resources directory if it doesn't exist
        os.makedirs('./resources', exist_ok=True)
        
        # Save the content and questions separately, without timing summary
        with open('./resources/raw_lesson_content.txt', 'w', encoding='utf-8') as f:
            f.write(full_content)
            
        with open('./resources/questions.txt', 'w', encoding='utf-8') as f:
            f.write(questions_content)
        
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

def count_tokens(text: str) -> int:
    """Count the number of tokens in the text using GPT tokenizer."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def main():
    parser = argparse.ArgumentParser(description='Generate educational content about a topic')
    parser.add_argument('--topic', type=str, required=True,
                      help='Topic to explore and generate content about')
    
    args = parser.parse_args()
    
    print(f"Initializing TopicExplorer...")
    explorer = TopicExplorer()
    
    print(f"Generating content about: {args.topic}")
    content = explorer.generate_full_content(args.topic)
    
    # Calculate content statistics
    char_length = len(content)
    token_count = count_tokens(content)
    
    print("\nContent Statistics:")
    print(f"Character Length: {char_length}")
    print(f"Token Count: {token_count}")

if __name__ == "__main__":
    main() 