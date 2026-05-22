from fastapi import FastAPI, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, IngestResponse
from app.rag.retriever import answer_query
from app.rag.ingest import ingest_documents

app = FastAPI(
    title="MAS Regulation Assistant",
    description="RAG chatbot for MAS regulatory documents",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "MAS-RAG-CHATBOT"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = answer_query(request.question)
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"]
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    try:
        ingest_documents()
        return IngestResponse(status="success", message="Documents ingested successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))