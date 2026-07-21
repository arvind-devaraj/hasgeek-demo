import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# ReAct = interleave REASONing ("Thought") with ACTing (a tool call), and feed
# each action's result ("Observation") back in so the next Thought can use it.
SYSTEM_PROMPT = """Solve the user's question using a Thought -> Action -> Observation loop.

TOOL:
- get_weather: {"city": "string"} -> current weather for a city.

Respond with exactly one JSON object per turn:
{"thought": "...", "action": {"name": "get_weather", "parameters": {"city": "..."}}}
Once you know the answer, respond with:
{"thought": "...", "action": {"name": "final_answer", "parameters": {"response": "..."}}}
"""


WEATHER_BY_CITY = {
    "tokyo": "18°C, Rainy",
    "bangalore": "22°C, Sunny",
    "london": "14°C, Overcast",
    "new york": "26°C, Humid",
}


def get_weather(city: str) -> str:
    return WEATHER_BY_CITY.get(city.lower().strip(), "20°C, Partly Cloudy")



def react_loop(user_query: str, max_steps: int = 4):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for _ in range(max_steps):
        # REASON: ask the model to think, then choose one action.
        reply = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
        ).choices[0].message.content

        data = json.loads(reply)
        print(json.dumps(data, indent=2))
        messages.append({"role": "assistant", "content": reply})


        action = data["action"]
        #print(f"Thought: {data['thought']}")

        if action["name"] == "final_answer":
            print(f"Answer: {action['parameters']['response']}")
            return

        # ACT: execute the tool the model chose.
        observation = get_weather(**action["parameters"])
        print(f"Action: get_weather({action['parameters']}) -> Observation: {observation}\n")

        # Feed the OBSERVATION back into the conversation for the next Thought.
        messages.append({"role": "user", "content": f"Observation: {observation}"})


if __name__ == "__main__":
    react_loop("What's the weather like in Tokyo and Bangalore?")
