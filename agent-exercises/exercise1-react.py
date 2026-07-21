import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =====================================================================
# 1. THE RE-ENGINEERED SYSTEM PROMPT
# =====================================================================
# We explicitly demand a valid, parsable JSON object for the Action block.
SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
You must process queries using a Thought -> Action -> Observation loop.

AVAILABLE TOOLS:
- get_weather: {"city": "string"} -> Returns current weather.

For every turn, you MUST output a single valid JSON object containing exactly two keys: "thought" and "action".
The "action" key must contain "name" (the tool name) and "parameters" (arguments object), OR "name": "final_answer" with a "response" string.

CRITICAL: Do not output any conversational text before or after the JSON block.

"""

# =====================================================================
# 2. MOCK TOOLS & SYSTEM ENVIRONMENT
# =====================================================================
def get_weather(city: str) -> str:
    city_lower = city.lower().strip()
    if "tokyo" in city_lower:
        return "18°C, Rainy"
    return "22°C, Sunny"

class OpenAILLM:
    """Calls the OpenAI Chat Completions API, using JSON mode to enforce the strict single-JSON-object output the system prompt requires."""
    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        return response.choices[0].message.content

# =====================================================================
# 3. THE STRUCTURED REACT LOOP
# =====================================================================
def run_structured_loop(user_query: str):
    llm = OpenAILLM()
    
    # We use a structured message array (like OpenAI/Ollama Chat API)
    # This keeps system instructions entirely separated from conversation history
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    print(f"User Query: {user_query}\n")
    
    for iteration in range(1, 4):
        print(f"--- Iteration {iteration} ---")
        
        # Step 1: LLM generates a text response (which should be strict JSON)
        raw_output = llm.generate(messages)
        print(raw_output)
        
        # Append the assistant's output to keep the conversation history tracked
        messages.append({"role": "assistant", "content": raw_output})
        
        # Step 2: Parse JSON safely
        try:
            response_data = json.loads(raw_output)
            #print(f"Thought: {response_data.get('thought')}")
            action = response_data.get("action", {})
            action_name = action.get("name")
            params = action.get("parameters", {})
        except json.JSONDecodeError:
            print("[Error] Small model failed to output valid JSON. Aborting.")
            break

        # Step 3: Branch based on Action Name
        if action_name == "final_answer":
            # The model occasionally puts "response" directly on the action
            # object instead of nesting it under "parameters" as instructed;
            # accept either shape rather than aborting on a minor deviation.
            final_response = params.get("response") or action.get("response")
            print(f"\nFinal Answer: {final_response}")
            break
            
        elif action_name == "get_weather":
            city_arg = params.get("city", "")
            print(f"[Tool Call] Executing get_weather for target: '{city_arg}'")
            
            # Execute actual python logic
            tool_output = get_weather(city_arg)
            print(f"[Observation] Tool returned: {tool_output}\n")
            
            # Feed the observation back to the model as a new user message
            # Labeling it clearly so the small model understands it came from the system
            messages.append({
                "role": "user", 
                "content": f"Observation from tool '{action_name}': {tool_output}"
            })
            
        else:
            print(f"[Error] Unknown tool action requested: {action_name}")
            break

if __name__ == "__main__":
    run_structured_loop("What's the weather like in Tokyo?")