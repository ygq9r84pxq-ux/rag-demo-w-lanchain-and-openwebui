# Local RAG Demo API

This project implements a Retrieval-Augmented Generation (RAG) system that allows a Large Language Model (xAI's Grok) to answer questions based on private, local PDF documents. It serves as an OpenAI-compatible backend for frontends like Open WebUI.

## 🌟 Features
- **Hybrid Search:** Prioritizes local documents (PDFs) and falls back to Tavily Web Search if no local data is found.
- **Persistent Memory:** Stores chat history in a PostgreSQL database, organized by User Session.
- **Smart Context:** Automatically summarizes conversation history when it gets too long to save tokens and cost.
- **Privacy First:** Uses local HuggingFace embeddings (`all-mpnet-base-v2`) running on CPU/GPU, ensuring document vectors stay private.
- **Streaming:** Supports token-by-token streaming response to the frontend.

## 🛠️ Prerequisites

### 1. System Requirements
- Python 3.10+
- PostgreSQL Database with `pgvector` extension installed.
- Ubuntu / Linux environment (recommended).

### 2. Environment Variables
Create a `.env` file in the root directory:
```bash
XAI_API_KEY="your-grok-api-key"
TAVILY_API_KEY="your-tavily-api-key"
```
### 3. Database Setup

Ensure your Postgres instance is running and accessible. The script requires two connection strings (one for ```psycopg2``` and one for ```psycopg``` v3), but they point to the same DB.

- Extension: You must run ```CREATE EXTENSION vector;``` in your database before running the script.

## 📦 Installation
Install the required Python packages:
```bash
pip install fastapi uvicorn sse-starlette pydantic python-dotenv
pip install langchain langchain-community langchain-huggingface langchain-postgres
pip install langchain-xai langchain-tavily
pip install psycopg[binary] pgvector sentence-transformers
```
## 🚀 Usage
#### 1. Ingest Documents (One-time setup)

Note: This logic is currently in the Jupyter Notebook ```rag_notebook.ipynb``` (not included in ```server.py```). You must ingest your PDFs into the local_docs vector collection before starting the server.

#### 2. Start the Server
```bash
python3 server.py
```

- The server runs on ```http://0.0.0.0:8000```.

- Swagger UI is available at ```http://0.0.0.0:8000/docs```.

#### 3. Connect Open WebUI

- Go to Open WebUI Settings > Connections.

- Set OpenAI API URL to ```http://<YOUR_SERVER_IP>:8000/v1```.

- Set API Key to any string (e.g., ```sk-123```).

- Refresh the models list. You should see ```local-rag```.

## ⚠️ Caveats & Known Issues (Not Production Ready)
This is a Proof of Concept (PoC) demo. It has the following limitations:

#### Security:

- There is no API Key validation on the ```/v1/chat/completions``` endpoint. Anyone with access to the port can query the model.
- CORS is set to ```allow_origins=["*"]```, which is permissive.

#### Concurrency:

- The current implementation uses a basic ```uvicorn``` setup. High traffic might require a production-grade WSGI/ASGI server (e.g., Gunicorn with Uvicorn workers).

#### Data Ingestion:

- The ```server.py``` does not handle document uploading. You must ingest documents using a separate script or notebook.

- If the PDF content is messy, the retrieval quality drops.

#### Error Handling:

- If the LLM or Database times out, the error messages returned to the frontend are basic.

## 🔮 Future Improvements for Production
To make this production-grade, the following features are needed:

- [ ] Authentication: Implement Bearer Token validation to secure the API.

- [ ] Admin UI: A dedicated interface to upload/delete PDFs without running Python scripts.

- [ ] Reranking: Add a "Reranker" step (e.g., Cohere Rerank or Cross-Encoder) to improve the accuracy of retrieved documents before sending them to the LLM.

- [ ] Citations: Update the prompt/response format to return strict JSON citations (Page #, Source File) so the UI can display clickable references.

- [ ] Vector DB Migration: Move from ```langchain_community.PGVector``` (deprecated) to the official ```langchain_postgres.PGVectorStore```.
