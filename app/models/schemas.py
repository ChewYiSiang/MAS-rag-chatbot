from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "example": {"question": "What are the AML requirements for banks in Singapore?"}
        }
    }


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int


class IngestResponse(BaseModel):
    status: str
    message: str