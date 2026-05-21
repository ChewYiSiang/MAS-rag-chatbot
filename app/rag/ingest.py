import os
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

DOCS_PATH = "data/docs"
CHROMA_PATH = "data/chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# flow: read PDF -> split into 500 characters chunk size with 50 character overlap -> each chunk is embedded into vectors by sentence-transformer model -> vectors stored in ChromaDB
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection(client):
    return client.get_or_create_collection(
        name="finrag",
        metadata={"hnsw:space": "cosine"}
    )

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# chunk overlaps prevent chunks that were together, from losing context
def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# embedding converts documents into numeric vectors that contain semantic meaning
def ingest_documents():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    client = get_chroma_client()
    collection = get_collection(client)

    pdf_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in {DOCS_PATH}. Add some MAS PDFs and try again.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(DOCS_PATH, pdf_file)
        print(f"Processing: {pdf_file}")

        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        print(f" -> {len(chunks)} chunks created")

        embeddings = model.encode(chunks).tolist()

        ids = [f"{pdf_file}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": pdf_file, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        print(f" -> stored in ChromaDB")

    print(f"\nDone. {len(pdf_files)} document(s) ingested.")


if __name__ == "__main__":
    ingest_documents()