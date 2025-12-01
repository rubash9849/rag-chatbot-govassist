from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os
import socket

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_all_data
from src.embedding import load_embedding_model, generate_embeddings
from src.vectorstore import init_vectorstore, add_documents_to_store
from src.search import init_gemini_client, rag_advanced

# Check and clear port 8000
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

if is_port_in_use(8000):
    print("⚠️  Port 8000 is in use. Attempting to clear...")
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.net_connections():
                    if conn.laddr.port == 8000:
                        print(f"Killing process {proc.pid} using port 8000")
                        proc.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
    except ImportError:
        print("⚠️  Install psutil to auto-clear ports: pip install psutil")
        print("Or manually kill the process using port 8000")

# Global variables
collection = None
model = None
client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG components on startup - OPTIMIZED to check existing vectorstore"""
    global collection, model, client
    
    print("🚀 Initializing backend...")
    try:
        # Find data directory - check multiple locations
        possible_paths = [
            "../data",        # If running from backend folder (most common)
            "data",           # If running from root
            "../../data"      # If running from nested folder
        ]
        
        data_path = None
        for path in possible_paths:
            # Check if path exists AND has the required subfolders
            if os.path.exists(path):
                text_exists = os.path.exists(os.path.join(path, "text_data"))
                json_exists = os.path.exists(os.path.join(path, "json_data"))
                pdf_exists = os.path.exists(os.path.join(path, "ssn_pdf"))
                
                if text_exists or json_exists or pdf_exists:
                    data_path = path
                    print(f"✅ Found data directory at: {os.path.abspath(path)}")
                    break
        
        if data_path is None:
            # Create in parent directory (root)
            print("⚠️  Data directory not found. Creating structure in root...")
            data_path = "../data"
            os.makedirs(os.path.join(data_path, "text_data"), exist_ok=True)
            os.makedirs(os.path.join(data_path, "json_data"), exist_ok=True)
            os.makedirs(os.path.join(data_path, "ssn_pdf"), exist_ok=True)
            print(f"✅ Created data directories at: {os.path.abspath(data_path)}")
            print("   Please add your documents and restart.")
        
        # ═══════════════════════════════════════════════════════
        # OPTIMIZED: Check if vectorstore already exists
        # ═══════════════════════════════════════════════════════
        model = load_embedding_model()
        embedding_dim = model.get_sentence_embedding_dimension()
        
        # Path to vectorstore
        vector_store_path = os.path.join(data_path, "vector_store")
        index_path = os.path.join(vector_store_path, "faiss.index")
        metadata_path = os.path.join(vector_store_path, "metadata.pkl")
        
        # Check if vectorstore exists
        vectorstore_exists = os.path.exists(index_path) and os.path.exists(metadata_path)
        
        if vectorstore_exists:
            # ✅ LOAD EXISTING VECTORSTORE (Fast!)
            print("="*60)
            print("✅ EXISTING VECTORSTORE FOUND - Loading from disk...")
            print("="*60)
            
            index, metadata, persist_dir = init_vectorstore(
                persist_directory=vector_store_path,
                embedding_dim=embedding_dim
            )
            
            collection = {
                "index": index,
                "metadata": metadata,
                "persist_dir": persist_dir
            }
            
            print(f"✅ Loaded vectorstore with {index.ntotal} documents")
            print("="*60)
            print("💡 TIP: To rebuild vectorstore, delete data/vector_store/ folder")
            print("="*60)
            
        else:
            # ❌ NO VECTORSTORE - Create new one
            print("="*60)
            print("⚠️  NO EXISTING VECTORSTORE - Building from scratch...")
            print("="*60)
            
            # Load all documents
            docs = load_all_data(base_path=data_path, split=True)
            
            if len(docs) == 0:
                print("⚠️  No documents found. Add files to data/ folders and restart.")
                print("   - data/text_data/ for .txt files")
                print("   - data/json_data/ for .json files")
                print("   - data/ssn_pdf/ for .pdf files")
                # Create empty collection
                index, metadata, persist_dir = init_vectorstore(
                    persist_directory=vector_store_path,
                    embedding_dim=embedding_dim
                )
                collection = {
                    "index": index,
                    "metadata": metadata,
                    "persist_dir": persist_dir
                }
            else:
                # Generate embeddings
                texts = [d.page_content for d in docs]
                print(f"Generating embeddings for {len(texts)} documents...")
                embeddings = generate_embeddings(texts, model)
                
                # Initialize and populate vectorstore
                index, metadata, persist_dir = init_vectorstore(
                    persist_directory=vector_store_path,
                    embedding_dim=embedding_dim
                )
                
                index, metadata = add_documents_to_store(
                    index,
                    metadata,
                    persist_dir,
                    docs,
                    embeddings
                )
                
                collection = {
                    "index": index,
                    "metadata": metadata,
                    "persist_dir": persist_dir
                }
                
                print("="*60)
                print(f"✅ Created new vectorstore with {index.ntotal} documents")
                print(f"📁 Saved to: {os.path.abspath(vector_store_path)}")
                print("="*60)
        
        # Initialize Gemini client
        client = init_gemini_client()
        print("✅ Backend ready!")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise - allow API to start but return errors on queries
    
    yield
    
    # Cleanup on shutdown
    print("🛑 Shutting down...")

# Initialize FastAPI with lifespan
app = FastAPI(title="RAG Chatbot API", version="1.0", lifespan=lifespan)

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    conversation_history: list = []
    enable_web_search: bool = True
    top_k: int = 8                      # ✅ ADDED - Receives from frontend
    score_threshold: float = 0.25       # ✅ ADDED - Receives from frontend

class ChatResponse(BaseModel):
    answer: str
    sources: list
    confidence: float
    used_web_search: bool

# Health check
@app.get("/health")
async def health():
    """Health check endpoint"""
    if collection is None:
        return {
            "status": "initializing",
            "message": "Backend is starting up...",
            "vectorstore_ready": False
        }
    
    return {
        "status": "healthy",
        "documents": collection["index"].ntotal if collection else 0,
        "vectorstore_ready": True
    }

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for RAG queries"""
    
    if collection is None or model is None or client is None:
        raise HTTPException(
            status_code=503,
            detail="Backend is still initializing. Please wait..."
        )
    
    try:
        # Call RAG pipeline with parameters from frontend
        result = rag_advanced(
            query=request.query,
            collection=collection,
            model=model,
            client=client,
            top_k=request.top_k,                        # ✅ ADDED - Uses slider value
            score_threshold=request.score_threshold,    # ✅ ADDED - Uses slider value
            enable_web_search=request.enable_web_search,
            conversation_history=request.conversation_history
        )
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            used_web_search=result.get("used_web_search", False)
        )
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Config endpoint
@app.get("/config")
async def get_config():
    """Get current RAG configuration"""
    from src.search import get_config_info
    return get_config_info()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)