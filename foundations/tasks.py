import ollama
import json

# Configuration
MODEL = "gemma3:4b"

def run_gemma_task(task_name, prompt, system_prompt="You are a helpful NLP assistant."):
    print(f"\n--- Running Task: {task_name} ---")
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
        )
        print(response['message']['content'])
    except Exception as e:
        print(f"Error: Ensure Ollama is running and {MODEL} is pulled. {e}")

# 1. Named Entity Recognition (NER)
ner_text = "Apple CEO Tim Cook visited Mumbai on April 18, 2026."
run_gemma_task("NER", f"Extract (Person, Org, Location, Date) as JSON from: {ner_text}")


# 2. Intent Classification
user_msg = "I need to reset my password, I forgot it."
run_gemma_task("Intent Classification", f"Classify this into [Support, Sales, Billing]: {user_msg}")
