from smolagents import ToolCallingAgent, PythonInterpreterTool, LiteLLMModel

model = LiteLLMModel(model_id="gpt-4o-mini")

# Create a traditional JSON-based tool-calling agent with access to Python code execution
agent = ToolCallingAgent(
    tools=[PythonInterpreterTool()],
    model=model
)

result = agent.run("What is the 50th Fibonacci number? Compute it using Python.")
print(result)