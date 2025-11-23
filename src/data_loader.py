import os
import glob
from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    JSONLoader,
    PyMuPDFLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------- TEXT LOADER ----------------
def load_text_data(base_path: str = "../data") -> List[Document]:
    text_path = os.path.join(base_path, "text_data")
    text_files = glob.glob(os.path.join(text_path, "*.txt"))

    docs = []
    for file_path in text_files:
        loader = TextLoader(file_path, encoding="utf-8")
        docs.extend(loader.load())

    print(f"Loaded {len(docs)} text documents.")
    return docs


# ---------------- JSON LOADER ----------------
def load_json_data(base_path: str = "../data") -> List[Document]:
    json_path = os.path.join(base_path, "json_data")
    json_files = glob.glob(os.path.join(json_path, "*.json"))

    all_docs = []
    for file_path in json_files:
        loader = JSONLoader(
            file_path=file_path,
            jq_schema="""
            (
              .sections[] | {heading: .heading, paragraphs: .paragraphs, list_items: .list_items, link: .link}
            ),
            (
              .subpages[]?.sections[] | {heading: .heading, paragraphs: .paragraphs, list_items: .list_items, link: .link}
            )
            """,
            text_content=False,
        )
        docs = loader.load()
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} JSON documents from {len(json_files)} files.")
    return all_docs


# ---------------- PDF LOADER ----------------
def load_pdf_data(base_path: str = "../data") -> List[Document]:
    pdf_path = os.path.join(base_path, "ssn_pdf")
    loader = DirectoryLoader(
        pdf_path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
    )
    pdf_docs = loader.load()
    print(f"Loaded {len(pdf_docs)} PDF documents.")
    return pdf_docs


# ---------------- SPLIT DOCUMENTS ----------------
def split_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


# ---------------- LOAD EVERYTHING ----------------
def load_all_data(base_path: str = "../data", split: bool = True) -> List[Document]:
    """
    Load all documents (text, json, pdf) and optionally split them into chunks.
    """
    text_docs = load_text_data(base_path)
    json_docs = load_json_data(base_path)
    pdf_docs = load_pdf_data(base_path)

    all_docs = text_docs + json_docs + pdf_docs
    print(f"Total combined documents: {len(all_docs)}")

    if split:
        all_docs = split_documents(all_docs)

    return all_docs
