import json
import inspect
import os
from typing import Dict, Callable, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =====================================================================
# 1. THE TOOLS REGISTRY
# =====================================================================
# Define our real Python application functions

def get_weather(city: str) -> str:
    """Gets the current weather metrics for a specified city."""
    city_lower = city.lower().strip()
    if "tokyo" in city_lower:
        return "18°C, Light Rain"
    elif "london" in city_lower:
        return "14°C, Overcast"
    return "21°C, Clear Skies"

def calculate_flight_time(origin: str, destination: str) -> str:
    """Calculates estimated travel duration between two international airports."""
    return f"Estimated flight duration from {origin} to {destination} is 11 hours and 30 minutes."

# The Central Registry: This maps string identifiers to live Python function objects.
TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "get_weather": get_weather,
    "calculate_flight_time": calculate_flight_time
}

# =====================================================================
# 2. AUTO-GENERATED SYSTEM PROMPT
# =====================================================================
def generate_system_prompt(registry: Dict[str, Callable]) -> str:
    """
    Dynamically generates structural instructions based on what
    tools are loaded into the registry using Python reflection (inspect).
    """
    tool_descriptions = ""
    for name, func in registry.items():
        # Read the function signature dynamically
        sig = inspect.signature(func)
        # Formulate a clean description for the model's awareness
        tool_descriptions += f"- {name}{sig}: {func.__doc__}\n"

    return f"""You are an autonomous operations agent. You solve user queries using a Thought -> Action -> Observation loop.

AVAILABLE TOOLS:
{tool_descriptions}
CRITICAL OUTPUT RULES:
1. Every response MUST be a single, raw, valid JSON object. Do not include markdown code blocks (like ```json).
2. The JSON object must contain exactly these two keys: "thought" and "action".
3. The "action" block must provide the tool "name" and its input "parameters" object.
4. When you have the final solution, set "name" to "final_answer" and put your response inside "parameters" under the key "response".

EXACT FORMAT TEMPLATE:
{{
  "thought": "Reasoning about what step to take next...",
  "action": {{
    "name": "tool_name_here",
    "parameters": {{"param_name": "value"}}
  }}
}}
"""

# =====================================================================
# 3. THE OPENAI REASONING ENGINE
# =====================================================================
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
# 4. AUTOMATED AGENT CORE ENGINE
# =====================================================================
class AutonomousAgentEngine:
    def __init__(self, tools: Dict[str, Callable], llm_client: Any):
        self.tools = tools
        self.llm = llm_client
        self.system_prompt = generate_system_prompt(tools)

    def execute(self, user_query: str, max_turns: int = 5) -> str:
        # Initialize context memory tracking array
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]

        print("=" * 60)
        print(f"🚀 Initializing Agent Engine for Query: '{user_query}'")
        print("=" * 60)

        for turn in range(1, max_turns + 1):
            print(f"\n[Turn {turn}] Consulting Reasoning Engine...")

            # Step 1: Call LLM with active context history
            raw_response = self.llm.generate(messages)

            # Commit the LLM's raw generation to internal dialogue history
            messages.append({"role": "assistant", "content": raw_response})

            # Step 2: Auto-parse the generated content
            try:
                data = json.loads(raw_response.strip())
                thought = data.get("thought", "Thinking...")
                action = data.get("action", {})
                tool_name = action.get("name")
                tool_params = action.get("parameters", {})

                print(f"🤖 Agent Thought: \"{thought}\"")
            except json.JSONDecodeError:
                print("❌ [Execution Error] Model generated unparsable layout. Forcing retry instruction.")
                messages.append({"role": "user", "content": "Error: Your last output was not valid JSON. Correct your format."})
                continue

            # Step 3: Evaluate Terminal State Condition
            if tool_name == "final_answer":
                print("\n🎯 [Terminal State Reached]")
                # The model occasionally puts "response" directly on the action
                # object instead of nesting it under "parameters", or nests a
                # structured object instead of a plain string; normalize all
                # of these rather than losing the answer to a minor deviation.
                final_response = tool_params.get("response") or action.get("response", "No answer extracted.")
                if not isinstance(final_response, str):
                    final_response = json.dumps(final_response)
                return final_response

            # Step 4: Dynamic Tool Discovery & Invocation Execution
            print(f"⚙️  [Action Detected] Requesting tool execution: `{tool_name}`")

            if tool_name in self.tools:
                # Resolve function pointer dynamically out of our registry dict map
                target_function = self.tools[tool_name]

                try:
                    # Execute tool via keyword expansions (**tool_params) safely
                    print(f"🔌 Calling Python environment runtime `{tool_name}` arguments -> {tool_params}")
                    observation = target_function(**tool_params)

                except TypeError as e:
                    observation = f"Execution Error: Argument mismatch for `{tool_name}`. Details: {e}"
            else:
                observation = f"Execution Error: Requested tool `{tool_name}` does not exist in the environment registry."

            print(f"👁️  [Observation Received] -> {observation}")

            # Step 5: Feed the observation directly back into context window for next turn
            messages.append({
                "role": "user",
                "content": f"Observation from runtime environment execution of '{tool_name}': {observation}"
            })

        return "Engine Timeout: Maximum loop limit reached without resolving a final answer."

# =====================================================================
# 5. RUNNING THE LIVE AGENT
# =====================================================================
if __name__ == "__main__":
    # Instantiate the real OpenAI-backed client and engine config
    llm_client = OpenAILLM()
    agent_engine = AutonomousAgentEngine(tools=TOOL_REGISTRY, llm_client=llm_client)

    # Run the automated process
    final_output = agent_engine.execute("I need to fly from Tokyo to London. How long is the flight and what's the weather like there?")

    print("\n" + "="*60)
    print(f"🎉 Final Agent Output delivered to User:\n{final_output}")
    print("="*60)
