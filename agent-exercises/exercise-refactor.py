import os
import json
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. INITIALIZE CLIENT
# Ensure OPENAI_API_KEY is set (via .env or the environment)
client = OpenAI()

# 2. DEFINE THE BASIC TOOLS
def read_file(path: str) -> str:
    """Reads and returns the full contents of a file."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(path: str, content: str) -> str:
    """Writes or overwrites content to a file."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def grep(pattern: str, path: str = ".") -> str:
    """Simulates a basic grep command searching for a regex pattern in text files."""
    results = []
    # Simple recursive directory walk ignoring hidden folders
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    for i, line in enumerate(f, 1):
                        if re.search(pattern, line):
                            results.append(f"{file_path}:{i}: {line.strip()}")
            except Exception:
                continue
    return "\n".join(results) if results else "No matches found."

# Tool mapping directory
TOOL_MAP = {
    "read_file": read_file,
    "write_file": write_file,
    "grep": grep
}

# JSON-schema tool declarations for the OpenAI Chat Completions API.
# Unlike the Gemini SDK, OpenAI can't derive a schema from a raw Python
# function/docstring, so each tool is declared explicitly here.
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads and returns the full contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes or overwrites content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write."},
                    "content": {"type": "string", "description": "Content to write to the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Recursively searches text files under a directory for a regex pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for."},
                    "path": {"type": "string", "description": "Directory to search in. Defaults to the current directory."},
                },
                "required": ["pattern"],
            },
        },
    },
]

# =====================================================================
# 3. SETUP MOCK WORKSPACE FOR THE DEMO
# =====================================================================
# Original ("before") file, kept intact so it can be diffed against
# whatever the agent produces. The agent is instructed to write its
# refactored version alongside it, not overwrite it in place.
BEFORE_PATH = "src/user_service_before.py"
AFTER_PATH = "src/user_service_after.py"
CONFIG_PATH = "config.json"

def setup_demo_environment():
    # Clean up any artifacts left over from a previous run so "before" vs
    # "after" always reflects this run, not a stale one.
    for stale_path in (AFTER_PATH, CONFIG_PATH):
        if os.path.exists(stale_path):
            os.remove(stale_path)

    user_service_before = """
import requests

def get_user_data(user_id):
    # TODO: Secure this endpoint
    url = f"https://api.production-server.com/v1/users/{user_id}"
    headers = {"Authorization": "Bearer super-secret-admin-token-123"}
    return requests.get(url, headers=headers).json()

def check_system_status():
    status_url = "https://api.production-server.com/v1/status"
    return requests.get(status_url).status_code
"""
    write_file(BEFORE_PATH, user_service_before.strip())

    main_app = f"""
from src.user_service_before import get_user_data

print("Starting App...")
data = get_user_data(42)
print("User Data fetched.")
"""
    write_file("main.py", main_app.strip())
    print(f"[Demo Setup] Mock codebase created: '{BEFORE_PATH}' has hardcoded URLs and secrets.\n")

# =====================================================================
# 4. THE AGENT LOOP
# =====================================================================
def run_agent(goal: str):
    system_instruction = f"""
    You are an elite, autonomous software engineering agent. Your objective is to achieve the user's goal using ONLY the provided tools.

    You operate in a strict loop:
    1. THOUGHT: Reason about what you need to do, what you just learned, and what tool to use next.
    2. ACTION: Call exactly ONE tool with its required arguments.
    3. OBSERVATION: You will receive the output of that tool, then you repeat.

    CRITICAL RULES:
    - Do NOT modify '{BEFORE_PATH}' — it must remain exactly as it is so it can be compared against your refactored version.
    - Write your refactored version of that file to '{AFTER_PATH}' instead.
    - Extract hardcoded URLs/secrets into a new file at '{CONFIG_PATH}'.
    - Execute actions sequentially. Do not stop until the goal is fully accomplished. When completely done, summarize your changes.
    """

    print(f"🚀 Agent started with Goal: '{goal}'\n" + "="*50)

    # Initialize conversation. Unlike Gemini's stateful `chat` object, the
    # OpenAI Chat Completions API is stateless per-call, so we maintain the
    # full trajectory ourselves and resend it every turn.
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": goal},
    ]

    # Loop max 10 times to prevent infinite runaways during a demo
    for turn in range(10):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
            temperature=0.2,  # Low temperature for reliable tool use
        )
        msg = response.choices[0].message

        # Print the Agent's reasoning if available
        if msg.content:
            print(f"\n🤔 Agent Thought:\n{msg.content.strip()}\n")

        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # Check if the model wants to call a function
        if not msg.tool_calls:
            print("🏁 Agent has concluded its work.")
            break

        for call in msg.tool_calls:
            tool_name = call.function.name
            try:
                tool_args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            print(f"🛠️  Executing Tool: {tool_name}({tool_args})")

            # Execute the function locally
            if tool_name in TOOL_MAP:
                try:
                    observation = TOOL_MAP[tool_name](**tool_args)
                except TypeError as e:
                    observation = f"Error: invalid arguments for {tool_name}: {e}"
            else:
                observation = f"Error: Tool {tool_name} not found."

            print(f"👁️  Observation Output:\n{observation}\n" + "-"*30)

            # Send the observation back to the agent as a "tool" message,
            # tagged with the tool_call_id it's answering.
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(observation),
            })
    else:
        print("🛑 Reached the max turn limit without the agent concluding.")

# =====================================================================
# 5. SHOW THE BEFORE/AFTER DIFF
# =====================================================================
def print_before_after():
    print("\n" + "=" * 50)
    print(f"📄 BEFORE ({BEFORE_PATH}):")
    print("-" * 50)
    print(read_file(BEFORE_PATH))
    print("\n" + "-" * 50)
    print(f"📄 AFTER ({AFTER_PATH}):")
    print("-" * 50)
    print(read_file(AFTER_PATH))
    print("=" * 50)

if __name__ == "__main__":
    setup_demo_environment()

    demo_goal = (
        "Find all hardcoded production URLs ('api.production-server.com') and secrets in "
        f"'{BEFORE_PATH}'. Extract them into '{CONFIG_PATH}', and write a refactored version of "
        f"the file that reads from that config, saved to '{AFTER_PATH}'."
    )

    run_agent(demo_goal)
    print_before_after()
