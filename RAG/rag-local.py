import ollama
import numpy as np

# 1. Our "Knowledge Base" (The Documents)
documents = [
    "The capital of Mars is Bradbury City, founded in 2031.",
    "Llamas are members of the camelid family and live in the Andes.",
    "The secret ingredient in the cookies is a pinch of sea salt.",
    "Gemma is a family of lightweight, state-of-the-art open models from Google."
]

def get_embedding(text):
    """Generate a vector for a given text using Embedding-Gemma."""
    response = ollama.embed(model='embeddinggemma', input=text)
    return np.array(response['embeddings'][0])

# 2. Indexing: Pre-calculate embeddings for our documents
doc_embeddings = [get_embedding(doc) for doc in documents]

def retrieve(query, top_k=1):
    """Find the most relevant document based on cosine similarity."""
    query_vec = get_embedding(query)
    
    # Calculate similarities
    similarities = [
        np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        for doc_vec in doc_embeddings
    ]
    
    # Return the best matching document
    best_idx = np.argmax(similarities)
    return documents[best_idx]

# 3. Execution: Question -> Retrieval -> Generation
query = "Where is the capital of Mars?"
context = retrieve(query)

prompt = f"Using only this context: {context}\n\nQuestion: {query}\nAnswer:"

response = ollama.generate(model='gemma3:4b', prompt=prompt)

print(f"--- RAG System ---\n")
print(f"Query: {query}")
print(f"Retrieved Context: {context}")
print(f"Final AI Answer: {response['response']}")