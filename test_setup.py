# test_setup.py
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

print("1. Testing ChromaDB...")
client = chromadb.Client()
col = client.create_collection("test")
col.add(documents=["MAS regulates financial institutions in Singapore."], ids=["1"])
results = col.query(query_texts=["what does MAS do?"], n_results=1)
print(f"   Retrieved: {results['documents'][0][0][:40]}")  # Print the first 40 characters of the retrieved document

print("2. Testing sentence-transformers...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("test sentence")
print(f"   Embedding shape: {embedding.shape}")

print("3. Testing Ollama...")
response = ollama.chat(model="llama3", messages=[
    {"role": "user", "content": "Reply with exactly: setup works"}
])
print(f"   Ollama says: {response['message']['content'][:40]}")

print("\n✓ Everything is working. You are ready to build.")
