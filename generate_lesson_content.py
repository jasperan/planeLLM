from transformers import pipeline
import torch

SYSTEM_PROMPT = """You are an expert educational content creator. Your task is to generate 
approximately 30 minutes worth of spoken content (about 4500 words) on a specific topic.

The content should be:
1. Well-structured with clear main points
2. Educational but accessible
3. Include relevant examples and analogies
4. Cover both basic and advanced concepts
5. Include potential questions a student might ask

Format the output as a detailed outline with main points and supporting details.
Do not include any formatting symbols or section numbers - just pure content.
"""

def generate_lesson_content(topic, model_name="meta-llama/Llama-2-7b-chat-hf"):
    """
    Generate educational content on a specific topic.
    
    Args:
        topic (str): The topic to generate content about
        model_name (str): Name of the model to use
    
    Returns:
        str: Generated educational content
    """
    pipeline = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    prompt = f"{SYSTEM_PROMPT}\n\nCreate educational content about: {topic}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    output = pipeline(
        messages,
        max_new_tokens=6000,  # Approximately 4500 words
        temperature=0.7,
        do_sample=True,
    )

    # Save the raw content
    content = output[0]["generated_text"][-1]['content']
    with open('./resources/raw_lesson_content.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    
    return content

if __name__ == "__main__":
    # Example usage
    topic = "Understanding Neural Networks: From Basics to Deep Learning"
    content = generate_lesson_content(topic)
    print("Content generated and saved to ./resources/raw_lesson_content.txt") 