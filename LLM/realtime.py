from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads OPENAI_API_KEY from .env
client = OpenAI()

# No tools, no web search, no system date injected — a plain model call like
# this can only draw on what it learned during training. It has no clock and
# no live connection to the outside world, so it cannot know today's date.
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "what is today's date."}],
)

print(response.choices[0].message.content)