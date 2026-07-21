from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel

# 1. Setup lightweight model (OpenAI, Claude, local, etc via LiteLLM)
model = LiteLLMModel(model_id="gpt-4o-mini")

# 2. Initialize CodeAgent with built-in search tool
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model
)

# 3. Execute query
result = agent.run(
    "Search for the distance between Earth and Mars today, "
    "and calculate how long light takes to travel that distance in minutes."
)
print(result)