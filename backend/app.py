from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.data_loader import load_all_data
from src.embedding import load_embedding_model, generate_embeddings
from src.vectorstore import init_vectorstore, add_documents_to_store
from src.search import init_gemini_client, rag_advanced

# Initialize FastAPI
app = FastAPI(title="RAG Chatbot API", version="1.0")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # use specific URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RAG components once
print("Initializing backend components...")
docs = load_all_data(base_path="data", split=True)
texts = [d.page_content for d in docs]
model = load_embedding_model()
embeddings = generate_embeddings(texts, model)
_, collection = init_vectorstore()
add_documents_to_store(collection, docs, embeddings)
client = init_gemini_client()


# Request schema
class ChatRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    """Handle chat messages from frontend."""
    result = rag_advanced(
        query=req.query,
        collection=collection,
        model=model,
        client=client,
        top_k=req.top_k,
        score_threshold=0.1,
        return_context=True,
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
    }
