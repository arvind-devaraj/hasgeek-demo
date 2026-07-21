import asyncio
import os

from dotenv import load_dotenv
from browser_use import Agent
from browser_use.llm import ChatOpenAI

load_dotenv()

# ChatBrowserUse (the browser-use hosted proxy model) requires its own
# BROWSER_USE_API_KEY from https://cloud.browser-use.com — we don't have one,
# but we do have OPENAI_API_KEY, so use browser-use's own OpenAI wrapper
# instead. It talks directly to the OpenAI API, same key as the other
# exercises in this workshop.
llm = ChatOpenAI(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))

agent = Agent(
    task="""
    Find a trip for 2 adults to Tokyo for 5 days next month:
    1. Search Google Flights for direct flights from SFO under $1,200 total.
    2. Find top 3 hotels on Booking.com in Shinjuku rated > 8.5 with a budget under $200/night.
    3. Check real-time weather predictions for those dates.
    4. Create an actionable markdown itinerary matching rainy/sunny days to indoor/outdoor activities.
    """,
    llm=llm,
)


async def main():
    # Cap the run so a demo can't silently spin for the library's default
    # of 500 steps if the agent gets stuck on a captcha or a dead end.
    result = await agent.run(max_steps=25)
    print(result.final_result())


if __name__ == "__main__":
    asyncio.run(main())
