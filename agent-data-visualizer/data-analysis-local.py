# Same task as data-analysis.py, but the code the model writes actually runs
# right here in this process — no OpenAI-hosted sandbox/container, no
# network round trip to execute code, no cold-start latency to reuse.

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; the model's code runs in
                        # this same process, off the main thread, where
                        # macOS's default GUI backend crashes.

from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel

load_dotenv()

agent = CodeAgent(
    model=OpenAIServerModel("gpt-4o-mini"),
    tools=[],
    additional_authorized_imports=["pandas", "seaborn", "matplotlib", "matplotlib.pyplot", "sklearn", "sklearn.datasets"],
)

result = agent.run(
    "Load the iris dataset from sklearn.datasets, clean any missing data, compute the "
    "correlation between features, and save a Seaborn heatmap plot as iris_heatmap.png "
    "in the current directory."
)
print(result)
