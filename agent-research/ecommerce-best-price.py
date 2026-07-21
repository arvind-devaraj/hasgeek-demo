from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel

load_dotenv()

# 1. Initialize the search tool that the agent can use to look up current prices
search_tool = DuckDuckGoSearchTool()

# 2. Define our AI model
model = OpenAIServerModel("gpt-4o-mini")

# 3. Create the agent and provide it with the search tool
agent = CodeAgent(
    tools=[search_tool], 
    model=model,
    additional_authorized_imports=["math", "re"] # Authorized for processing numeric string patterns
)

# 4. Define the prompt containing your conditional discount logic
prompt = (
    "Search for the current price of a 128GB iPhone 15 on Amazon India. "
    "Once you have found the price, calculate the final billing amount using these rules: "
    "1. Calculate a 10% instant discount based on the retrieved price. "
    "2. If the discount amount is greater than ₹1,500, cap the discount at exactly ₹1,500. "
    "3. Subtract the discount from the original price to find the final billing amount. "
    "Print out the initial price, the calculated discount applied, and your final billing amount."
)

# 5. Execute the agent
response = agent.run(prompt)
print(response)