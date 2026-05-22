from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "MAS-RAG-CHATBOT"}


def test_chat_empty_question():
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_chat_returns_answer():
    mock_result = {
        "answer": "Banks must conduct customer due diligence under MAS Notice 626.",
        "sources": ["MAS Notice 626 Amendment new.pdf"],
        "chunks_used": 4
    }
    with patch("app.main.answer_query", return_value=mock_result):
        response = client.post("/chat", json={"question": "What are AML requirements?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "chunks_used" in data
        assert len(data["answer"]) > 0


def test_chat_response_structure():
    mock_result = {
        "answer": "MAS requires financial institutions to maintain AML policies.",
        "sources": ["MAS Notice 626 Amendment new.pdf"],
        "chunks_used": 3
    }
    with patch("app.main.answer_query", return_value=mock_result):
        response = client.post("/chat", json={"question": "Tell me about MAS policies"})
        data = response.json()
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["chunks_used"], int)