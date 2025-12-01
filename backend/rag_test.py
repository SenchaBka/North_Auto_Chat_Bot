# Core FastAPI application instance and startup.
import re
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.build_vectorstore import build_vectorstore_from_argument
build_vectorstore_from_argument("Dealer for volvo is at volvostreet 123")
from services.genai_service import build_graph



graph = build_graph()
class RAGRequest(BaseModel):
    query: str

# Config required for checkpointer - provides session/thread ID
config = {"configurable": {"thread_id": "test_session"}}

while True: 
    result = graph.invoke({"query": input("Enter your query: "), "messages": []}, config=config)
    print(result)