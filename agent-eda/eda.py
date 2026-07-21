import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# 1. Load messy data
df = pd.read_csv("customer_churn.csv")

# 2. Spin up the Pandas Agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

# 3. Ask complex visual/analytical questions
agent.invoke("Find top 3 reasons customers churn and plot a bar chart comparing them by monthly charges.")