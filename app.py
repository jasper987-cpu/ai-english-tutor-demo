import gradio as gr
from transformers import pipeline

# Load the existing LLM from Hugging Face (free)
pipe = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)

def chat(message, history):
    # Simple system prompt for English tutor
    full_prompt = f"""You are a friendly English communication tutor.
    Help the user with grammar, vocabulary, stories, or conversation practice.
    User: {message}"""
    
    result = pipe(full_prompt, max_new_tokens=300, temperature=0.7, do_sample=True)
    response = result[0]["generated_text"].split("User:")[-1].strip()
    return response

demo = gr.ChatInterface(
    chat,
    title="🤖 My First English Tutor (Powered by Llama 3.1)",
    description="Type anything — grammar correction, vocabulary, story, conversation practice",
    theme="soft"
)

demo.launch()