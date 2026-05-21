import ollama
from sentence_transformers import SentenceTransformer
from app.rag.ingest import get_chroma_client, get_collection, EMBED_MODEL

TOP_K = 4
MODEL = "llama3"

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model

# same process like ingesting of documents, user query is being chunked and embedded into vectors so that it can be searched in vector database
def retrieve_chunks(query: str) -> list[dict]:
    model = get_embed_model()
    query_embedding = model.encode(query).tolist()

    client = get_chroma_client()
    collection = get_collection(client)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "score": round(1 - results["distances"][0][i], 3)
        })

    return chunks

def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"Source: {c['source']}\n{c['text']}" for c in chunks
    )
    return f"""You are a regulatory assistant specializing in MAS (Monetary Authority of Singapore) regulations.
    
    Answer the question using ONLY the context provided below. If the answer is not in the context, say "I could not find this in the provided MAS documents."
    
    Context: {context}
    Question: {query}

    Answer:"""

def answer_query(query: str) -> dict:
    chunks = retrieve_chunks(query)

    if not chunks:
        return {
            "answer": "No documents have been ingested yet. Please run the ingestion first.",
            "sources": []
        }
    
    prompt = build_prompt(query, chunks)

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    sources = list({c["source"] for c in chunks})

    return {
        "answer": response["message"]["content"],
        "sources": sources,
        "chunks_used": len(chunks)
    }

if __name__ == "__main__":
    print("Testing retriever...")
    result = answer_query("What are the AML requirements for banks in Singapore?")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")