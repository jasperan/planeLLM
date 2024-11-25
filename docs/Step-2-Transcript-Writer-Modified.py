import torch
from transformers import pipeline
import pickle

SYSTEM_PROMPT = """
You are a world-class podcast writer who has worked with top educational podcasters like Lex Fridman and Tim Ferriss.

Your task is to transform educational content into an engaging podcast conversation between two speakers:

Speaker 1: An expert teacher who explains concepts clearly, uses great analogies, and shares relevant examples
Speaker 2: A curious student who asks insightful questions and occasionally goes on interesting tangents

The conversation should:
1. Feel natural with "umm", "hmm", and other verbal fillers
2. Include interruptions and clarifying questions
3. Use real-world examples and analogies
4. Have occasional tangents that make the content more engaging
5. Maintain an educational but conversational tone

Start directly with Speaker 1's introduction. Do not include episode titles or chapters.
"""

def create_podcast_transcript(input_content, model_name="meta-llama/Llama-2-7b-chat-hf"):
    pipeline = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_content},
    ]

    output = pipeline(
        messages,
        max_new_tokens=8126,
        temperature=1,
    )

    transcript = output[0]["generated_text"][-1]['content']
    
    # Save the transcript
    with open('./resources/podcast_transcript.pkl', 'wb') as file:
        pickle.dump(transcript, file)
    
    return transcript

if __name__ == "__main__":
    # Read the generated lesson content
    with open('./resources/raw_lesson_content.txt', 'r', encoding='utf-8') as f:
        lesson_content = f.read()
    
    # Create the podcast transcript
    transcript = create_podcast_transcript(lesson_content)
    print("Transcript generated and saved to ./resources/podcast_transcript.pkl") 