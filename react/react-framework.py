from dotenv import load_dotenv
from smolagents import OpenAIServerModel, ToolCallingAgent, tool

load_dotenv()

# Same ReAct loop as react-loop.py / react-toolcall.py — REASON, choose an
# ACTION, feed the OBSERVATION back in — but the loop itself (message list,
# tool_calls parsing, re-prompting) is no longer our code at all. The
# framework's ToolCallingAgent runs it internally; we only supply the tool.


WEATHER_BY_CITY = {
    "tokyo": "18°C, Rainy",
    "bangalore": "22°C, Sunny",
    "london": "14°C, Overcast",
    "new york": "26°C, Humid",
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city to check.
    """
    return WEATHER_BY_CITY.get(city.lower().strip(), "20°C, Partly Cloudy")


agent = ToolCallingAgent(tools=[get_weather], model=OpenAIServerModel("gpt-4o-mini"))

if __name__ == "__main__":
    print(agent.run("What's the weather like in Tokyo and Bangalore?"))
