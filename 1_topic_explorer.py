import oci
import yaml
import json
from typing import List, Dict
import time
import os

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
        
        # Llama 3.1 70B model ID
        self.model_id = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyaiir6nnhmlgwvh37dr2mvragxzszqmz3hok52pcgmpqta"

    def generate_questions(self, topic: str) -> List[str]:
        """Generate relevant questions about the topic."""
        prompt = f"""You are an expert researcher. Generate 8-10 specific, detailed questions about {topic}.
        The questions should:
        1. Cover different aspects and time periods
        2. Include both well-known and lesser-known facts
        3. Focus on interesting historical events, people, and places
        4. Be specific enough to generate detailed responses
        
        Format: Return only the questions, one per line."""

        response = self._call_llm(prompt)
        questions = [q.strip() for q in response.split('\n') if q.strip() and '?' in q]
        return questions

    def explore_question(self, question: str) -> str:
        """Generate detailed content for a specific question."""
        prompt = f"""As an expert storyteller, provide a detailed response to this question:
        {question}
        
        Your response should:
        1. Be detailed and engaging (aim for 500-700 words)
        2. Include specific dates, names, and places
        3. Share interesting anecdotes or lesser-known facts
        4. Connect events to their broader context
        """
        
        return self._call_llm(prompt)

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
        print(f"Generating questions about {topic}...")
        questions = self.generate_questions(topic)

        print('Questions: {}'.format(questions))
        
        full_content = f"# {topic}\n\n"
        
        for i, question in enumerate(questions, 1):
            print(f"Exploring question {i}/{len(questions)}: {question}")
            response = self.explore_question(question)
            full_content += f"\n## {question}\n\n{response}\n\n"
            print('Generated {} tokens'.format(len(response)))
        
        # Create resources directory if it doesn't exist
        os.makedirs('./resources', exist_ok=True)
        
        # Save the content
        with open('./resources/raw_lesson_content.txt', 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        return full_content

if __name__ == "__main__":
    explorer = TopicExplorer()
    
    # Example usage
    topic = "Fort Worth and its history"
    content = explorer.generate_full_content(topic)
    print("\nContent generated and saved to ./resources/raw_lesson_content.txt") 