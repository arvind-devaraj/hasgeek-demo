import ollama

# Single text embedding
response = ollama.embed(
    model='embeddinggemma',
    input='The quick brown fox jumps over the lazy dog.'
)

# Extract the vector
embedding = response['embeddings'][0]
print(f"Vector Length: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")