# basic RAG with no framework — just the OpenAI SDK and numpy, so the
# actual mechanism (embed, compare, retrieve, generate) is visible instead
# of hidden behind llama_index's VectorStoreIndex/query_engine abstractions.

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

kb_array = [
    "The 2025 Tokyo Protocol mandates a global coal phase-out by 2035.",
    "NASA's Artemis IV landed a pressurized rover on Mars in January 2026.",
    "The March 2026 AI Act requires digital watermarking on all AI media.",
    "The SF 49ers won Super Bowl LX in February 2026 against the Chiefs.",
]


def embed(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([item.embedding for item in response.data])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b)
    return a_norm @ b_norm


def query(question: str, top_k: int = 2) -> str:
    # RETRIEVE: embed the knowledge base and the question, then rank
    # documents by cosine similarity to find the most relevant ones.
    doc_embeddings = embed(kb_array)
    query_embedding = embed([question])[0]

    similarities = cosine_similarity(doc_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    retrieved_docs = [kb_array[i] for i in top_indices]

    # AUGMENT + GENERATE: stuff the retrieved context into the prompt and
    # ask the model to answer using only that context.
    context = "\n".join(retrieved_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer using only the context above."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("\n--- Response ---")
    print(query("What are the new AI rules for 2026?"))
