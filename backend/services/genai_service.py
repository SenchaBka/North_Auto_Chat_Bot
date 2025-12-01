# Handles all Gemini API calls (chat logic, safe prompting).
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import TypedDict, List, Annotated, Literal, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import operator


def load_vectorstore(index_name: str = "faiss_index"):
    """
    Loads a FAISS vectorstore by name.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    index_path = os.path.join(backend_dir, index_name)
    
    if not os.path.exists(index_path):
        raise RuntimeError(f" FAISS index '{index_name}' not found. Run: python build_vectorstore.py")
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectorstore = FAISS.load_local(
        folder_path=index_path,
        embeddings=embedder,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Load FAISS retrievers for different document sets
car_info_vectorstore = load_vectorstore("faiss_car_info_index")
car_info_retriever = car_info_vectorstore.as_retriever(search_kwargs={"k": 3})
dealer_info_vectorstore = load_vectorstore("faiss_dealer_info_index")
dealer_info_retriever = dealer_info_vectorstore.as_retriever(search_kwargs={"k": 3})

# Document set registry - maps document set names to their retrievers
DOCUMENT_SETS = {
    "car_info": {
        "retriever": car_info_retriever,
        "description": "Information about cars, car models, specifications, and automotive details"
    },
    "dealer_info": {
        "retriever": dealer_info_retriever,
        "description": "Information about car dealerships"
    }
}


# --------------------- Tools ---------------------------
@tool
def search_car_info(query: str) -> str:
    """
    Search the car information database for details about vehicles,
    car models, specifications, features, and automotive information.
    Use this tool when the user asks about cars, vehicles, or automotive topics.
    """
    docs = DOCUMENT_SETS["car_info"]["retriever"].invoke(query)
    if not docs:
        return "No relevant car information found."
    return "\n\n".join([f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in docs])


@tool
def search_dealer_info(query: str) -> str:
    """
    Search the dealer information database for details about car dealerships.
    Use this tool for questions not specifically about cars or vehicles.
    """
    docs = DOCUMENT_SETS["dealer_info"]["retriever"].invoke(query)
    if not docs:
        return "No relevant information found."
    return "\n\n".join([f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in docs])


@tool
def get_available_document_sets() -> str:
    """
    List all available document sets that can be searched.
    Use this when the user asks what information sources are available.
    """
    result = "Available document sets:\n"
    for name, info in DOCUMENT_SETS.items():
        result += f"- {name}: {info['description']}\n"
    return result


# All available tools
TOOLS = [search_car_info, search_dealer_info, get_available_document_sets]


# --------------------- State ---------------------------
def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Reducer that appends new messages to existing message history."""
    return left + right


class RAGState(TypedDict):
    query: str
    context: str
    answer: str
    source_docs: List[str]
    # Chat memory - accumulates messages across turns
    messages: Annotated[List[BaseMessage], add_messages]
    # Tool-related state
    selected_tool: Optional[str]
    tool_output: Optional[str]


# --------------------- Prompts -------------------------
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a routing assistant. Based on the user's query and conversation history,
determine which tool to use:
- "search_car_info": For questions about cars, vehicles, models, specifications
- "search_dealer_info": For questions about car dealerships
- "get_available_document_sets": When user asks what sources are available
- "none": For simple greetings or when no search is needed

Respond with ONLY the tool name, nothing else."""),
    ("user", "Conversation history:\n{history}\n\nCurrent query: {query}")
])
ROUTER_CHAIN = ROUTER_PROMPT | llm | StrOutputParser()

GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant for North Auto. You help users with car-related questions.
Use the provided context and conversation history to give accurate, helpful responses.
If the context doesn't contain relevant information, say so honestly."""),
    ("user", """Conversation history:
{history}

Context from documents:
{context}

Current question: {query}

Provide a helpful answer:""")
])
GEN_CHAIN = GEN_PROMPT | llm | StrOutputParser()


# --------------------- Helper Functions -----------------
def format_message_history(messages: List[BaseMessage], max_messages: int = 10) -> str:
    """Format message history for prompts, limiting to recent messages."""
    if not messages:
        return "No previous conversation."
    
    recent_messages = messages[-max_messages:]
    formatted = []
    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
    
    return "\n".join(formatted) if formatted else "No previous conversation."


# --------------------- Nodes ---------------------------
def route_query_node(state: RAGState) -> RAGState:
    """Determine which tool to use based on the query and conversation history."""
    history = format_message_history(state.get("messages", []))
    
    tool_choice = ROUTER_CHAIN.invoke({
        "query": state["query"],
        "history": history
    }).strip().lower()
    
    # Validate tool choice
    valid_tools = ["search_car_info", "search_dealer_info", "get_available_document_sets", "none"]
    if tool_choice not in valid_tools:
        tool_choice = "search_car_info"  # Default to car info search
    
    return {
        **state,
        "selected_tool": tool_choice,
    }


def execute_tool_node(state: RAGState) -> RAGState:
    """Execute the selected tool and retrieve context."""
    selected_tool = state.get("selected_tool", "none")
    query = state["query"]
    
    tool_output = ""
    sources = []
    
    if selected_tool == "search_car_info":
        tool_output = search_car_info.invoke(query)
        sources = ["car_info_database"]
    elif selected_tool == "search_dealer_info":
        tool_output = search_dealer_info.invoke(query)
        sources = ["dealer_info_database"]
    elif selected_tool == "get_available_document_sets":
        tool_output = get_available_document_sets.invoke("")
        sources = ["system"]
    else:
        tool_output = "No specific context needed for this query."
        sources = []
    
    return {
        **state,
        "context": tool_output,
        "tool_output": tool_output,
        "source_docs": sources,
    }


def generate_node(state: RAGState) -> RAGState:
    """Generate response using context and conversation history."""
    history = format_message_history(state.get("messages", []))
    
    answer = GEN_CHAIN.invoke({
        "query": state["query"],
        "context": state.get("context", ""),
        "history": history
    })
    
    return {
        **state,
        "answer": answer,
    }


def update_memory_node(state: RAGState) -> RAGState:
    """Update chat memory with the current turn."""
    new_messages = [
        HumanMessage(content=state["query"]),
        AIMessage(content=state["answer"])
    ]
    
    return {
        **state,
        "messages": new_messages,
    }


# --------------------- Build Graph ----------------------
def build_graph():
    """Build the RAG graph with chat memory and tool-use capabilities."""
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("route_query", route_query_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("generate", generate_node)
    graph.add_node("update_memory", update_memory_node)
    
    # Set entry point
    graph.set_entry_point("route_query")
    
    # Add edges
    graph.add_edge("route_query", "execute_tool")
    graph.add_edge("execute_tool", "generate")
    graph.add_edge("generate", "update_memory")
    graph.add_edge("update_memory", END)
    
    # Compile with memory checkpointer for conversation persistence
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# --------------------- Session Management ---------------
class ChatSessionManager:
    """Manages chat sessions with persistent memory."""
    
    def __init__(self):
        self.graph = build_graph()
    
    def chat(self, query: str, session_id: str = "default") -> dict:
        """
        Process a chat message with session-based memory.
        
        Args:
            query: The user's message
            session_id: Unique identifier for the conversation session
            
        Returns:
            Dictionary with answer, sources, and tool used
        """
        config = {"configurable": {"thread_id": session_id}}
        
        result = self.graph.invoke(
            {"query": query, "messages": []},
            config=config
        )
        
        return {
            "answer": result.get("answer"),
            "sources": result.get("source_docs", []),
            "tool_used": result.get("selected_tool"),
        }
    
    def get_history(self, session_id: str = "default") -> List[dict]:
        """
        Get the conversation history for a session.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            List of message dictionaries with role and content
        """
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            state = self.graph.get_state(config)
            messages = state.values.get("messages", [])
            
            history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"role": "assistant", "content": msg.content})
            
            return history
        except Exception:
            return []


# Create a global session manager instance
session_manager = ChatSessionManager()
