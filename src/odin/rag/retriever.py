"""RAG retrieval using ChromaDB and Sentence Transformers.

This module provides the runtime retrieval interface for the ODIN framework.
It queries a pre-built ChromaDB collection to find relevant documentation,
examples, and API references given an error context.
"""
import logging
import time
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from odin.config import config
from odin.state import ErrorCategory, RetrievalRecord

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieves relevant knowledge from ChromaDB."""
    
    def __init__(self):
        """Initialize the retriever with embedding model and ChromaDB client."""
        self.embedding_model = SentenceTransformer(config.rag_embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(config.rag_db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get or create the collection
        try:
            self.collection = self.client.get_collection("odin_knowledge")
            logger.info(f"Connected to existing collection with {self.collection.count()} documents")
        except Exception as e:
            logger.warning(f"Collection not found or error: {e}. Creating empty collection.")
            self.collection = self.client.create_collection(
                name="odin_knowledge",
                metadata={"description": "Python documentation and code examples for ODIN"}
            )
    
    def retrieve(
        self,
        query: str,
        category: ErrorCategory,
        iteration: int,
        top_k: int | None = None,
    ) -> RetrievalRecord:
        """Retrieve relevant knowledge for an error.
        
        Args:
            query: The error message or traceback to search for
            category: The error category (used for metadata filtering if available)
            iteration: Current iteration number
            top_k: Number of results to return (defaults to config.rag_top_k)
        
        Returns:
            RetrievalRecord with retrieved chunks and metadata
        """
        start_time = time.perf_counter()
        top_k = top_k or config.rag_top_k
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()
        
        # Query ChromaDB
        # We can optionally filter by category if we tagged documents with categories
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        
        end_time = time.perf_counter()
        retrieval_latency_ms = (end_time - start_time) * 1000
        
        # Format results
        chunks = []
        total_tokens = 0
        
        if results["documents"] and results["documents"][0]:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunk_info = {
                    "content": doc,
                    "metadata": metadata or {},
                    "similarity_score": 1.0 - distance,  # Convert distance to similarity
                }
                chunks.append(chunk_info)
                
                # Approximate token count (rough heuristic: 1 token ≈ 4 chars)
                total_tokens += len(doc) // 4
        
        logger.info(
            f"Retrieved {len(chunks)} chunks for category '{category}' "
            f"in {retrieval_latency_ms:.2f}ms"
        )
        
        return RetrievalRecord(
            iteration=iteration,
            triggered=True,
            decision_reason=f"Retrieval triggered for category '{category}'",
            query=query,
            chunks=chunks,
            retrieval_latency_ms=retrieval_latency_ms,
            total_tokens=total_tokens,
        )
    
    def health_check(self) -> dict[str, Any]:
        """Check if the retriever is operational.
        
        Returns:
            Dictionary with health status and statistics
        """
        try:
            doc_count = self.collection.count()
            return {
                "status": "healthy",
                "document_count": doc_count,
                "collection_name": self.collection.name,
                "embedding_model": config.rag_embedding_model,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
