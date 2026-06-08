"""
FinSight — Embeddings & FAISS index management

Two public functions:
  build_index(pdf_paths, config)  → builds and saves a FAISS index from PDF files
  load_index(config)              → loads an existing FAISS index from disk
"""

from __future__ import annotations
from typing import Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import FinSightConfig


def _get_embeddings(config: FinSightConfig) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=config.embedding_model)


def build_index(
    pdf_paths: Dict[str, str],   # {"apple": "/path/apple_10k.pdf", "tesla": ...}
    config: FinSightConfig,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> FAISS:
    """
    Load PDFs, chunk them, embed with HuggingFace MiniLM, and save a FAISS index.

    Args:
        pdf_paths: mapping of company_key → pdf file path.
                   The company_key is stored as metadata["company"] for RAG filtering.
        config:    FinSightConfig instance.
        chunk_size / chunk_overlap: RecursiveCharacterTextSplitter settings.

    Returns:
        Compiled FAISS vectorstore (also saved to config.faiss_index_path).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    all_chunks = []

    for company_key, pdf_path in pdf_paths.items():
        print(f"  Loading {company_key} from {pdf_path} ...")
        loader = PyPDFLoader(pdf_path)
        docs   = loader.load()
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["company"] = company_key
        all_chunks.extend(chunks)
        print(f"    → {len(docs)} pages, {len(chunks)} chunks")

    print(f"\nTotal chunks to index: {len(all_chunks)}")
    embeddings   = _get_embeddings(config)
    vectorstore  = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(config.faiss_index_path)
    print(f"✅ FAISS index saved to {config.faiss_index_path}")
    return vectorstore


def load_index(config: FinSightConfig) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Returns:
        FAISS vectorstore ready for similarity search.
    """
    embeddings  = _get_embeddings(config)
    vectorstore = FAISS.load_local(
        config.faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"✅ FAISS index loaded from {config.faiss_index_path}")
    return vectorstore
