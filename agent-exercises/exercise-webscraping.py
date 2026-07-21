from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(model_id="gpt-4o-mini")

agent = CodeAgent(
    tools=[], 
    model=model,
    additional_authorized_imports=["httpx", "bs4", "json"]
)

# The agent writes the scraping and DOM selection logic in Python natively
result = agent.run(
    "Fetch the front page of Hacker News (https://news.ycombinator.com/), "
    "extract the titles and point counts for the top 5 stories, and return them as a JSON list."
)
print(result)