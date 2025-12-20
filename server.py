import os
import uuid
import time
import json
import urllib.parse
import psycopg
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Modern import
from langchain_community.vectorstores import PGVector
from langchain_xai import ChatXAI
from langchain_tavily import TavilySearch
from langchain_postgres import PostgresChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURATION ---
load_dotenv()  # Load keys from .env file

# Ensure keys exist
if not os.getenv("XAI_API_KEY"):
    raise ValueError("XAI_API_KEY is missing in .env file")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY is missing in .env file")

# DB Config - Update these with your actual credentials
DB_USER = "postgres"
DB_PASS = "postgresewer@#E32eWwWeqe333r"
DB_HOST = "192.168.10.10"
DB_PORT = "5432"
DB_NAME = "embeddings_db1"

encoded_pass = urllib.parse.quote_plus(DB_PASS)
# Standard connection string for PGVector (Psycopg2)
CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{encoded_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# Async-friendly connection string for History (Psycopg 3)
HISTORY_CONN_STRING = f"postgresql://{DB_USER}:{encoded_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- 2. GLOBAL OBJECTS ---
embeddings = None
vectorstore = None
retriever = None
llm = None
web_search_tool = None
sync_connection = None

# --- 3. LIFECYCLE MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize everything on startup
    global embeddings, vectorstore, retriever, llm, web_search_tool, sync_connection
    
    print("🚀 Starting RAG Server...")
    
    # A. Setup DB Connection for History
    sync_connection = psycopg.connect(HISTORY_CONN_STRING)
    PostgresChatMessageHistory.create_tables(sync_connection, "chat_history")
    
    # B. Setup Embeddings & VectorStore
    # Using local HuggingFace embeddings (Privacy first)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    vectorstore = PGVector(
        collection_name="local_docs",
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )
    
    # Retrieval Settings: k=3 prevents context flooding
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # C. Setup LLM & Tools
    llm = ChatXAI(model="grok-beta", temperature=0)
    web_search_tool = TavilySearch(max_results=3)
    
    yield
    
    # Cleanup on shutdown
    print("🛑 Shutting down...")
    if sync_connection:
        sync_connection.close()

app = FastAPI(title="Local RAG API", lifespan=lifespan)

# --- 4. CORS MIDDLEWARE (Crucial for Open WebUI) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to your UI's IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. HELPER FUNCTIONS ---

def ensure_uuid(id_str: str) -> str:
    """Ensures session_id is a valid UUID, generating a consistent one if not."""
    try:
        return str(uuid.UUID(id_str))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))

def get_session_history(session_id: str):
    valid_id = ensure_uuid(session_id)
    return PostgresChatMessageHistory("chat_history", valid_id, sync_connection=sync_connection)

def clear_session_history(session_id):
    history = get_session_history(session_id)
    history.clear()
    print(f"🧹 History wiped for {session_id}")

def manage_history(session_id):
    """Summarizes history if it exceeds a certain length to save tokens."""
    history = get_session_history(session_id)
    msgs = history.messages
    if len(msgs) > 6:
        print(f"⚠️ History too long for {session_id}, summarizing...")
        # Simple summarization chain
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the chat history into one concise paragraph."),
            ("placeholder", "{chat_history}")
        ])
        chain = summary_prompt | llm | StrOutputParser()
        summary = chain.invoke({"chat_history": msgs})
        
        history.clear()
        history.add_message(SystemMessage(content=f"SUMMARY OF PAST CONVERSATION: {summary}"))

# --- 6. API MODELS (OpenAI Compatible) ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "local-rag"
    messages: List[ChatMessage]
    stream: bool = True
    user: Optional[str] = "default_user"  # Open WebUI sends this!

# --- 7. ENDPOINTS ---

@app.get("/v1/models")
async def list_models():
    """Required for Open WebUI to discover this backend."""
    return {
        "object": "list",
        "data": [
            {
                "id": "local-rag",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "triagelogic"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 1. Extract Info
    user_query = request.messages[-1].content
    user_id = request.user if request.user else "default_user"
    
    # 2. Special Commands for Testing
    if "/clear" in user_query or "/reset" in user_query:
        clear_session_history(user_id)
        return {"choices": [{"message": {"role": "assistant", "content": "Memory cleared."}}]}

    # 3. RAG Pre-processing (History Management)
    manage_history(user_id)
    
    # A. Search Local Knowledge Base
    docs = retriever.invoke(user_query)
    context = ""
    source_label = "Local Docs"
    
    if docs:
        # Join doc content safely
        context = "\n\n".join([d.page_content for d in docs])
    else:
        # Fallback to Web Search
        source_label = "Web Search"
        print("🌍 Local docs empty, searching web...")
        web_res = web_search_tool.invoke({"query": user_query})
        # Tavily returns a list of dicts, convert to string
        context = str(web_res)

    # 4. Prompt Construction
    history = get_session_history(user_id).messages
    
    system_prompt = (
        "You are a helpful assistant for TriageLogic. "
        "Use the provided Context and History to answer. "
        "Strictly ignore prior history if it contradicts the current Context. "
        "If the answer is not in the context, say you don't know."
    )
    
    # We define the variables in the template but fill them in the chain call
    # ensuring no f-string injection happens here.
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "Context ({source_label}):\n{context}\n\nQuestion:\n{question}")
    ])
    
    # 5. Streaming Logic
    async def event_generator():
        chain = prompt | llm | StrOutputParser()
        full_answer = ""
        
        # Pass variables explicitly to avoid LangChain parsing errors
        async for chunk in chain.astream({
            "chat_history": history,
            "context": context,
            "source_label": source_label,
            "question": user_query
        }):
            full_answer += chunk
            # OpenAI Stream Format
            data = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(data)}\n\n"
        
        # Save to DB after streaming is done
        get_session_history(user_id).add_messages([
            HumanMessage(content=user_query),
            AIMessage(content=full_answer)
        ])
        
        # Send 'Done' signal
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Run on 0.0.0.0 to be accessible by Open WebUI
    uvicorn.run(app, host="0.0.0.0", port=8000)
