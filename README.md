# MAS FinReg Assistant 🏦

A production-grade RAG (Retrieval-Augmented Generation) chatbot that answers questions about MAS (Monetary Authority of Singapore) regulatory documents. Built to demonstrate end-to-end MLOps skills.
From RAG pipeline design to containerised deployment with CI/CD and Kubernetes orchestration.

---

## What it does

Financial compliance teams, analysts, and fintech developers spend significant time manually reading through MAS regulatory PDFs. This assistant lets you ask natural language questions and get accurate, source-cited answers grounded strictly in the official documents. No hallucinations. No guessing.

**Example queries:**
- *"What are the AML requirements for banks under MAS Notice 626?"*
- *"What customer due diligence obligations apply to payment service providers?"*
- *"What are the technology risk management requirements for financial institutions?"*

---

## Architecture

```
User Question
     │
     ▼
┌─────────────┐
│  FastAPI    │  REST API — /chat, /ingest, /health
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│           RAG Pipeline              │
│                                     │
│  1. Embed query (all-MiniLM-L6-v2)  │
│  2. Retrieve top-4 chunks           │
│     from ChromaDB                   │
│  3. Build prompt with context       │
│  4. Generate answer via Ollama      │
│     (Llama 3, runs locally)         │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  ChromaDB   │  Vector store — persistent embeddings
└─────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Llama 3 via Ollama (local, free) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |
| Vector Store | ChromaDB |
| PDF Parsing | PyPDF2 |
| Containerisation | Docker + Docker Compose |
| Orchestration | Kubernetes (minikube) |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## Project Structure

```
MAS-RAG-chatbot/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── rag/
│   │   ├── ingest.py           # PDF loading, chunking, embedding
│   │   ├── embed.py            # Embedding utilities
│   │   └── retriever.py        # Query, retrieve, generate
│   └── models/
│       └── schemas.py          # Pydantic request/response models
├── data/
│   └── docs/                   # MAS regulatory PDFs
├── tests/
│   ├── test_api.py             # API endpoint tests
│   └── test_ingest.py          # Ingestion pipeline tests
├── k8s/
│   ├── deployment.yaml         # Kubernetes deployments
│   └── service.yaml            # Kubernetes services
├── .github/workflows/
│   └── ci.yml                  # GitHub Actions CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/download) with Llama 3 pulled
- [Docker Desktop](https://www.docker.com/products/docker-desktop)

### 1. Clone and set up environment

```bash
git clone https://github.com/ChewYiSiang/MAS-RAG-chatbot.git
cd MAS-RAG-chatbot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add MAS documents

Download regulatory PDFs from [mas.gov.sg](https://www.mas.gov.sg/regulation) and place them in `data/docs/`.

### 3. Run with Docker Compose

```bash
docker-compose up --build
```

### 4. Ingest documents

Once containers are running, trigger ingestion via the API:

```bash
curl -X POST http://localhost:8000/ingest
```

Or visit **http://localhost:8000/docs** for the interactive API UI.

### 5. Ask a question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the AML requirements for banks in Singapore?"}'
```

**Response:**
```json
{
  "answer": "According to MAS Notice 626, banks must conduct Customer Due Diligence (CDD)...",
  "sources": ["MAS Notice 626 Amendment new.pdf"],
  "chunks_used": 4
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/chat` | Ask a question, get a sourced answer |
| `POST` | `/ingest` | Trigger document ingestion into ChromaDB |

Full interactive docs available at `/docs` when running.

---

## CI/CD Pipeline

Every push to `main` triggers the GitHub Actions pipeline:

```
Push to main
     │
     ▼
┌──────────────┐     fail → ✗ build blocked
│  Run pytest  │
└──────┬───────┘
       │ pass
       ▼
┌──────────────────┐
│ Build Docker img │
└──────┬───────────┘
       │
       ▼
┌─────────────────────┐
│ Push to Docker Hub  │
└─────────────────────┘
```

Tests cover: API endpoint responses, request validation, chunking logic, and mocked RAG responses to keep CI fast and dependency-free.

---

## Kubernetes Deployment

The app runs as two deployments on Kubernetes connected via ClusterIP services.:
- The API (2 replicas)
- ChromaDB (1 replica)

```bash
# Deploy to local minikube cluster
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify pods are running
kubectl get pods

# Scale the API horizontally
kubectl scale deployment finreg-api --replicas=3
```

### Live cluster screenshots

**Pods running (2 API replicas + ChromaDB):**

![kubectl get pods](screenshots/kubectl-get-pods.png)

**Services:**

![kubectl get services](screenshots/kubectl-get-services.png)

**Deployments:**

![kubectl get deployments](screenshots/kubectl-get-deployments.png)

![kubectl describe deployment](screenshots/kubectl-describe-deployment.png)

**Scaling to 3 replicas:**

![kubectl scale](screenshots/kubectl-scale.png)

![kubectl scale](screenshots/kubectl-after-scale.png)

**Cluster status:**

![minikube status](screenshots/minikube-status.png)

---

## How RAG works in this project

1. **Ingestion** — MAS PDFs are parsed, split into 500-character chunks with 50-character overlap, and embedded using `all-MiniLM-L6-v2`. Embeddings are stored in ChromaDB with source metadata.

2. **Retrieval** — The user's question is embedded using the same model. ChromaDB performs cosine similarity search to find the 4 most relevant chunks.

3. **Generation** — The retrieved chunks are assembled into a prompt with strict instructions to answer only from the provided context. Llama 3 generates the final answer locally via Ollama.

This grounding approach means the chatbot never fabricates regulatory information. If the answer isn't in the documents, it says so.

---

## Documents Included

- MAS Notice 626 — Anti-Money Laundering / Countering the Financing of Terrorism
- MAS Notice 824 — Guidelines on Risk Management
- MAS PSG-01 — Guidelines on Licensing for Payment Service Providers

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Author

**Chew Yi Siang**
[LinkedIn](https://www.linkedin.com/in/yi-siang-chew-617576212/) · [GitHub](https://github.com/ChewYiSiang) · [Portfolio](https://personal-portfolio-website-olive-five.vercel.app/)