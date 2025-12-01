"""
build_vectorstore.py
--------------------
Rebuilds the FAISS vectorstore used by the LangGraph RAG example.
Run this script when:
- You update data/info_cars.txt
- You add new documents
- The faiss_index folder is missing
Usage:
python build_vectorstore.py
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pyparsing import C
# -------------------------------------------------------------
# Load API key
# -------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError(" GOOGLE_API_KEY missing in .env")
# File paths (relative to backend folder)
# -------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
DATA_FILE = os.path.join(BACKEND_DIR, "data", "car_info.txt")
CAR_INFO_INDEX_PATH = os.path.join(BACKEND_DIR, "faiss_car_info_index")
DEALER_INFO_INDEX_PATH = os.path.join(BACKEND_DIR, "faiss_dealer_info_index")
# -------------------------------------------------------------
# Rebuild FAISS vectorstore
# -------------------------------------------------------------
def build_vectorstore():
    print(f" Building FAISS vectorstore from {DATA_FILE}...")
    # Ensure file exists
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f" Cannot find: {DATA_FILE}")
    # Read content
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
        if not text.strip():
            raise ValueError(" data/info.txt is empty!")
    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    chunks = splitter.split_text(text)
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "source": "info.txt",
                "chunk_id": i
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    print(f" Generated {len(docs)} chunks.")
    # Embeddings model
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=API_KEY
    )
    # Build FAISS
    db = FAISS.from_documents(docs, embedder)
    db.save_local(CAR_INFO_INDEX_PATH)
    print(f" Saved FAISS index to ./{CAR_INFO_INDEX_PATH}/")
    print("\n Done! You can now run:")
    print(" uvicorn main:app --reload --port 8000\n")
    
def build_vectorstore_from_argument(text: str):
    print(f" Building FAISS vectorstore from argument text...")
    if not text.strip():
        raise ValueError(" Argument text is empty!")
    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    chunks = splitter.split_text(text)
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "source": "argument_text",
                "chunk_id": i
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    print(f" Generated {len(docs)} chunks.")
    # Embeddings model
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=API_KEY
    )
    # Build FAISS
    db = FAISS.from_documents(docs, embedder)
    db.save_local(DEALER_INFO_INDEX_PATH)
    print(f" Saved FAISS index to ./{DEALER_INFO_INDEX_PATH}/")
    print("\n Done! You can now run:")
    print(" uvicorn main:app --reload --port 8000\n")
# -------------------------------------------------------------
# Execute
# -------------------------------------------------------------
if __name__ == "__main__":
    build_vectorstore()