# Core FastAPI application instance and startup.
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from services.genai_service import session_manager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class RAGRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"


class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    tool_used: Optional[str] = None


class HistoryResponse(BaseModel):
    messages: List[dict]


class SessionRequest(BaseModel):
    session_id: str = "default"


@app.post("/api/chat", response_model=RAGResponse)
def rag_endpoint(req: RAGRequest):
    """
    Chat endpoint with session-based memory and tool selection.
    
    - Maintains conversation history per session_id
    - Automatically selects appropriate tools based on query
    """
    result = session_manager.chat(
        query=req.query,
        session_id=req.session_id
    )
    return RAGResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        tool_used=result.get("tool_used")
    )


@app.post("/api/chat/history", response_model=HistoryResponse)
def get_chat_history(req: SessionRequest):
    """
    Get conversation history for a session.
    """
    history = session_manager.get_history(session_id=req.session_id)
    return HistoryResponse(messages=history)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)