import os

from dotenv import load_dotenv
from smolagents import OpenAIServerModel, ToolCallingAgent, tool

load_dotenv()

# =====================================================================
# 1. THE TOOL
# =====================================================================
# smolagents derives the JSON tool schema straight from the type hints
# and the Google-style docstring below (name, param types, descriptions) —
# this replaces the manual `generate_system_prompt()` reflection the
# hand-rolled version of this exercise used to build the same thing.
@tool
def check_flight_status(flight_number: str) -> str:
    """Checks the real-time status of a flight using its flight number identifier.

    Args:
        flight_number: The flight number to look up, e.g. "NH211".
    """
    fn = flight_number.upper().strip()
    if "NH211" in fn:
        return "Delayed by 2 hours due to incoming weather."
    elif "JL041" in fn:
        return "On Time. Scheduled departure from gate 4B."
    return "Flight status unknown. Please verify flight number."


# =====================================================================
# 2. THE AGENT
# =====================================================================
# ToolCallingAgent runs the same Thought -> Action(JSON) -> Observation
# loop that exercise3's MemoryManagedAgent built by hand, but state
# ("THE TRAJECTORY STATE MEMORY") is now handled internally by
# `agent.memory`: every step (task, tool call, tool result) is recorded
# there and the full history is resent to the model turn, which is what
# stops it from calling the same tool twice for no reason.
def build_agent() -> ToolCallingAgent:
    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    return ToolCallingAgent(tools=[check_flight_status], model=model, max_steps=5)


# =====================================================================
# 3. RUNTIME TEST INVOCATION
# =====================================================================
if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: Please set your OPENAI_API_KEY environment variable before running.")
        exit(1)

    agent = build_agent()

    query = "My flight is NH211. Can you look up its current status for me?"
    print("=" * 70)
    print(f"🎬 Initiating Run for Query: '{query}'")
    print("=" * 70)

    result = agent.run(query)

    print("\n" + "=" * 70)
    print(f"🎉 Final Response to User:\n{result}")
    print("=" * 70)

    # Surface the recorded trajectory state — the same thing exercise3's
    # `trajectory_state` list held, now built and owned by the framework.
    print("\n📜 Recorded trajectory (agent.memory):")
    for step in agent.memory.get_succinct_steps():
        if "task" in step:
            print(f"[Task] {step['task']}")
            continue
        for call in step.get("tool_calls") or []:
            fn = call["function"]
            print(f"[Step {step['step_number']}] Action: {fn['name']}({fn['arguments']})")
        print(f"[Step {step['step_number']}] Observation: {step.get('observations')}")
