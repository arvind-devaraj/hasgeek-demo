#basic RAG with llamaindex openai embedding

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

# LLM: The "Brain"
Settings.llm = OpenAI(model="gpt-4o-mini")

# EMBEDDING: The "Search"
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    embed_batch_size=100  # Sends multiple items in one go to save time
)

# 3. Simple Array Knowledge Base
kb_array = [
    "The 2025 Tokyo Protocol mandates a global coal phase-out by 2035.",
    "NASA's Artemis IV landed a pressurized rover on Mars in January 2026.",
    "The March 2026 AI Act requires digital watermarking on all AI media.",
    "The SF 49ers won Super Bowl LX in February 2026 against the Chiefs."
]

# 4. Build and Query
documents = [Document(text=item) for item in kb_array]

# This will now work without the 404 error
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

print("\n--- Response ---")
print(query_engine.query("What are the new AI rules for 2026?"))