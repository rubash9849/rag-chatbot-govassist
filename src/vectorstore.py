import os
import uuid
import numpy as np
import chromadb
from typing import List, Dict, Any


def init_vectorstore(
    persist_directory: str = "data/vector_store",
    collection_name: str = "rag_collection",
):
    """
    Initialize or load a ChromaDB persistent vector store.
    """
    os.makedirs(persist_directory, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"description": "RAG document embeddings"}
    )

    print(f"Vector store initialized: {collection_name}")
    print(f"Current stored documents: {collection.count()}")
    return client, collection


def add_documents_to_store(collection, documents: List[Any], embeddings: np.ndarray):
    """
    Add a list of documents and their embeddings to the vector store.
    """
    if len(documents) != len(embeddings):
        raise ValueError("Number of documents and embeddings must match.")

    print(f"Adding {len(documents)} documents to vector store...")

    ids = []
    metadatas = []
    texts = []

    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
        ids.append(doc_id)

        # Prepare metadata
        metadata = dict(doc.metadata)
        metadata["length"] = len(doc.page_content)
        metadata["index"] = i
        metadatas.append(metadata)

        # Store content
        texts.append(doc.page_content)

    # Add to Chroma
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=texts,
    )

    print(f"Successfully added {len(documents)} documents.")
    print(f"Total documents in store: {collection.count()}")


def query_vectorstore(
    collection, query_embedding: np.ndarray, top_k: int = 5
) -> Dict[str, Any]:
    """
    Query the vector store using a query embedding.
    Returns top_k matching documents and metadata.
    """
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    formatted_results = []
    for doc, meta, dist in zip(docs, metas, distances):
        formatted_results.append(
            {"content": doc, "metadata": meta, "similarity": 1 - dist}
        )

    print(f"Retrieved {len(formatted_results)} documents.")
    return formatted_results
