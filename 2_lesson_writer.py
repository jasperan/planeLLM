import oci
import yaml
import json
import pickle
import os
import time

class PodcastWriter:
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
        
        # Initialize timing dictionary
        self.execution_times = {
            'llm_calls': [],
            'total_time': 0
        }

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
        start_time = time.time()
        
        # Ensure resources directory exists
        os.makedirs('./resources', exist_ok=True)
        
        system_prompt = """You are a world-class podcast writer who has worked with top educational podcasters like Lex Fridman and Tim Ferriss.

        Your task is to transform educational content into an engaging podcast conversation between two speakers:

        Speaker 1: An expert historian who explains concepts clearly, uses great analogies, and shares relevant examples
        Speaker 2: A curious student who asks insightful questions and occasionally goes on interesting tangents

        The conversation should:
        1. Feel natural with "umm", "hmm", and other verbal fillers
        2. Include interruptions and clarifying questions
        3. Use real-world examples and analogies
        4. Have occasional tangents that make the content more engaging
        5. Maintain an educational but conversational tone

        Start directly with an introduction to the tool (planeLLM) that allows to create content like this. Do not include episode titles or chapters.

        Here's the content to transform:

        """
                
        # Combine prompts and make the API call
        full_prompt = system_prompt + input_content
        transcript = self._call_llm(full_prompt)
        
        # Save as text for easy reading
        with open('./resources/podcast_transcript.txt', 'w', encoding='utf-8') as file:
            file.write(transcript)
        
        # Store total execution time
        self.execution_times['total_time'] = time.time() - start_time
        
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

if __name__ == "__main__":
    print("Reading lesson content...")
    # Read the generated lesson content
    with open('./resources/raw_lesson_content.txt', 'r', encoding='utf-8') as f:
        lesson_content = f.read()
    
    print("\nInitializing PodcastWriter...")
    # Create the podcast transcript
    writer = PodcastWriter()
    
    print("\nGenerating podcast transcript...")
    transcript = writer.create_podcast_transcript(lesson_content)
    
    print("\nTranscript generated and saved to:")
    print("- ./resources/podcast_transcript.txt")
    
    # Print timing summary
    print("\n" + writer._generate_timing_summary()) 