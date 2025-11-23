import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.vectorstore import query_vectorstore
import google.genai as genai
from dotenv import load_dotenv


# ---------------- GEMINI CLIENT ----------------
def init_gemini_client():
    """
    Initialize Gemini API client from environment variable.
    Make sure .env contains GEMINI_API_KEY
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    client = genai.Client(api_key=api_key)
    print("Gemini client initialized.")
    return client


# ---------------- BASIC RAG SEARCH ----------------
def rag_simple(
    query: str,
    collection,
    model: SentenceTransformer,
    client,
    top_k: int = 3,
):
    """
    Retrieve top-k context chunks from vectorstore, send to Gemini, return concise answer.
    """
    print(f"RAG Search for: '{query}'")

    # Step 1: Generate query embedding
    query_embedding = model.encode([query])[0]

    # Step 2: Retrieve top chunks
    results = query_vectorstore(collection, query_embedding, top_k=top_k)
    context = "\n\n".join([doc["content"] for doc in results]) if results else ""

    if not context:
        return "No relevant context found."

    # Step 3: Build LLM prompt
    prompt = f"""
Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:
"""

    # Step 4: Query Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    # Step 5: Return answer
    answer = getattr(response, "text", "No answer generated.").strip()
    return answer


# ---------------- ADVANCED RAG SEARCH ----------------
def rag_advanced(
    query: str,
    collection,
    model: SentenceTransformer,
    client,
    top_k: int = 5,
    score_threshold: float = 0.1,
    return_context: bool = False,
) -> Dict[str, Any]:
    """
    Advanced RAG pipeline returning answer, sources, confidence, and optionally context.
    """
    print(f"Running advanced RAG pipeline for query: '{query}'")

    # Step 1: Embed query
    query_embedding = model.encode([query])[0]

    # Step 2: Retrieve documents
    results = query_vectorstore(collection, query_embedding, top_k=top_k)

    if not results:
        return {
            "answer": "No relevant context found.",
            "sources": [],
            "confidence": 0.0,
            "context": "",
        }

    # Step 3: Filter by threshold
    filtered_results = [r for r in results if r["similarity"] >= score_threshold]
    if not filtered_results:
        return {
            "answer": "No relevant documents above threshold.",
            "sources": [],
            "confidence": 0.0,
            "context": "",
        }

    # Step 4: Build context and sources
    context = "\n\n".join([r["content"] for r in filtered_results])
    sources = [
        {
            "source": r["metadata"].get("source", "unknown"),
            "score": r["similarity"],
            "preview": r["content"][:120] + "...",
        }
        for r in filtered_results
    ]

    confidence = max(r["similarity"] for r in filtered_results)

    # Step 5: Build Gemini prompt
    prompt = f"""
Use the following context to answer the question concisely and accurately.

Context:
{context}

Question: {query}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    output = {
        "answer": getattr(response, "text", "No answer generated.").strip(),
        "sources": sources,
        "confidence": confidence,
    }

    if return_context:
        output["context"] = context

    print("RAG answer generated successfully.")
    return output
