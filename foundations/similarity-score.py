import numpy as np
import ollama

def get_vec(text):
    """Fetch embedding vector from local Ollama instance using Gemma."""
    response = ollama.embeddings(model='embeddinggemma', prompt=text)
    return np.array(response['embedding'])

# Generate vectors
doc_vec = get_vec("A fluffy cat is sleeping.")
query_vec = get_vec("A resting feline.")

# Calculate Similarity (Dot Product)
# Note: For truer semantic similarity, consider Cosine Similarity
similarity = np.dot(doc_vec, query_vec)

print(f"Similarity Score: {similarity:.4f}")