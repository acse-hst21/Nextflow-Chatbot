# backend/app/rag_service.py
import os
import json
import pickle
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore

from .document_fetcher import fetch_nextflow_docs


class NextflowRAGService:
    """RAG service for Nextflow documentation using LangChain."""

    def __init__(
        self,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_store_path: str = "./vector_store",
        metadata_path: str = "./vector_store_metadata.json"
    ):
        """
        Initialize RAG service.

        Args:
            openai_api_key: OpenAI API key
            embedding_model: OpenAI embedding model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            vector_store_path: Path to persist vector store
            metadata_path: Path to store metadata about the vector store
        """
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_path = vector_store_path
        self.metadata_path = metadata_path

        # Initialize LangChain components
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=embedding_model
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.vector_store: Optional[VectorStore] = None
        self.last_updated: Optional[datetime] = None

        # Load existing vector store if available
        self._load_existing_vector_store()

    def _load_existing_vector_store(self) -> bool:
        """Load existing vector store from disk."""
        try:
            if os.path.exists(self.vector_store_path) and os.path.exists(self.metadata_path):
                # Load metadata
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)

                self.last_updated = datetime.fromisoformat(metadata.get('last_updated', ''))

                # Load FAISS vector store
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

                print(f"[{datetime.utcnow().isoformat()}] Loaded existing vector store from {self.vector_store_path}")
                print(f"[{datetime.utcnow().isoformat()}] Last updated: {self.last_updated}")
                return True

        except Exception as e:
            print(f"[{datetime.utcnow().isoformat()}] Error loading existing vector store: {e}")

        return False

    def _save_vector_store(self):
        """Save vector store and metadata to disk."""
        try:
            if self.vector_store:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)

                # Save FAISS vector store
                self.vector_store.save_local(self.vector_store_path)

                # Save metadata
                metadata = {
                    'last_updated': self.last_updated.isoformat() if self.last_updated else None,
                    'embedding_model': self.embedding_model,
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'document_count': self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0
                }

                with open(self.metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                print(f"[{datetime.utcnow().isoformat()}] Saved vector store to {self.vector_store_path}")

        except Exception as e:
            print(f"[{datetime.utcnow().isoformat()}] Error saving vector store: {e}")
            raise

    def _create_langchain_documents(self, raw_docs: List[Dict[str, Any]]) -> List[Document]:
        """Convert raw documents to LangChain Document objects with chunking."""
        documents = []

        for doc in raw_docs:
            # Split document into chunks
            chunks = self.text_splitter.split_text(doc["text"])

            for i, chunk in enumerate(chunks):
                # Create metadata for each chunk
                metadata = {
                    "source": doc["source_path"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "file_name": doc["file_name"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "doc_id": doc["id"],
                    "fetched_at": doc["fetched_at"]
                }

                # Create LangChain Document
                documents.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))

        return documents

    async def update_vector_store(self, force_update: bool = False) -> bool:
        """
        Update vector store with latest Nextflow documentation.

        Args:
            force_update: Force update even if recently updated

        Returns:
            True if update was performed, False if skipped
        """
        try:
            # Check if update is needed
            if not force_update and self.last_updated:
                time_since_update = datetime.utcnow() - self.last_updated
                if time_since_update < timedelta(days=7):
                    print(f"[{datetime.utcnow().isoformat()}] Vector store is recent (updated {time_since_update} ago), skipping update")
                    return False

            print(f"[{datetime.utcnow().isoformat()}] Starting vector store update...")

            # Fetch latest documents
            raw_docs = fetch_nextflow_docs()

            if not raw_docs:
                print(f"[{datetime.utcnow().isoformat()}] No documents fetched, aborting update")
                return False

            # Convert to LangChain documents with chunking
            documents = self._create_langchain_documents(raw_docs)

            print(f"[{datetime.utcnow().isoformat()}] Created {len(documents)} document chunks from {len(raw_docs)} source documents")

            # Create new vector store
            print(f"[{datetime.utcnow().isoformat()}] Creating embeddings and building vector store...")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)

            # Update timestamp
            self.last_updated = datetime.utcnow()

            # Save to disk
            self._save_vector_store()

            print(f"[{datetime.utcnow().isoformat()}] Vector store update completed successfully")
            return True

        except Exception as e:
            print(f"[{datetime.utcnow().isoformat()}] Error updating vector store: {e}")
            raise

    def search_documents(self, query: str, k: int = 3, score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using vector similarity.

        Args:
            query: Search query
            k: Number of documents to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of relevant documents with metadata
        """
        if not self.vector_store:
            print(f"[{datetime.utcnow().isoformat()}] No vector store available, returning empty results")
            return []

        try:
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k * 2)  # Get more than needed for filtering

            # Filter by score threshold and format results
            filtered_results = []
            for doc, score in results:
                # Convert distance to similarity (FAISS returns distance, lower is better)
                similarity_score = 1 / (1 + score)

                if similarity_score >= score_threshold:
                    result = {
                        "id": doc.metadata.get("doc_id", f"chunk_{doc.metadata.get('chunk_index', 0)}"),
                        "title": doc.metadata.get("title", "Unknown"),
                        "text": doc.page_content,
                        "url": doc.metadata.get("url", ""),
                        "source_path": doc.metadata.get("source", ""),
                        "file_name": doc.metadata.get("file_name", ""),
                        "similarity_score": round(similarity_score, 3),
                        "chunk_index": doc.metadata.get("chunk_index", 0),
                        "total_chunks": doc.metadata.get("total_chunks", 1)
                    }
                    filtered_results.append(result)

            # Limit to requested number of results
            return filtered_results[:k]

        except Exception as e:
            print(f"[{datetime.utcnow().isoformat()}] Error searching documents: {e}")
            return []

    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the current vector store."""
        info = {
            "vector_store_exists": self.vector_store is not None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "vector_store_path": self.vector_store_path
        }

        if self.vector_store and hasattr(self.vector_store, 'index'):
            info["document_count"] = self.vector_store.index.ntotal

        return info

    def is_update_needed(self) -> bool:
        """Check if vector store needs updating (older than 7 days)."""
        if not self.last_updated:
            return True

        time_since_update = datetime.utcnow() - self.last_updated
        return time_since_update >= timedelta(days=7)


# Global RAG service instance
_rag_service: Optional[NextflowRAGService] = None


def get_rag_service(openai_api_key: str) -> NextflowRAGService:
    """Get or create global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = NextflowRAGService(openai_api_key)
    return _rag_service


def initialize_rag_service(openai_api_key: str) -> NextflowRAGService:
    """Initialize RAG service and perform initial setup if needed."""
    global _rag_service
    _rag_service = NextflowRAGService(openai_api_key)

    # Check if initial indexing is needed
    if not _rag_service.vector_store or _rag_service.is_update_needed():
        print(f"[{datetime.utcnow().isoformat()}] Initial vector store setup needed")

    return _rag_service