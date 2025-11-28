"""
RAG (Retrieval Augmented Generation) Engine for Medical Chatbot.
Handles ChromaDB integration and document retrieval.
"""
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG engine for retrieving relevant medical documents from ChromaDB."""
    
    def __init__(
        self,
        chromadb_path: str = "../parquet cromadb",
        collection_name: str = "medical_documents",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG engine with ChromaDB.
        
        Args:
            chromadb_path: Path to ChromaDB directory
            collection_name: Name of the collection to use
            embedding_model: Sentence transformer model for embeddings
        """
        self.chromadb_path = chromadb_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.client = None
        self.collection = None
        
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            logger.info(f"Initializing ChromaDB from: {self.chromadb_path}")
            
            # Create ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.chromadb_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
            
            logger.info("ChromaDB client created successfully")
            
            # Initialize embedding function
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=embedding_function
                )
                doc_count = self.collection.count()
                logger.info(f"Loaded collection '{self.collection_name}' with {doc_count} documents")
            except Exception as e:
                logger.warning(f"Collection not found, attempting to list available collections: {e}")
                collections = self.client.list_collections()
                if collections:
                    # Use the first available collection
                    self.collection = self.client.get_collection(
                        name=collections[0].name,
                        embedding_function=embedding_function
                    )
                    logger.info(f"Using collection: {collections[0].name}")
                else:
                    raise Exception("No collections found in ChromaDB")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def retrieve_context(
        self,
        query: str,
        n_results: int = 3,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User's question
            n_results: Number of documents to retrieve
            min_similarity: Minimum similarity threshold (0-1)
        
        Returns:
            List of relevant documents with metadata
        """
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return []
            
            logger.info(f"Querying ChromaDB for: '{query[:50]}...'")
            
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results and results['documents'] and len(results['documents']) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
                distances = results['distances'][0] if results['distances'] else [0] * len(documents)
                
                for doc, metadata, distance in zip(documents, metadatas, distances):
                    # Convert distance to similarity score (lower distance = higher similarity)
                    similarity = 1 / (1 + distance)
                    
                    if similarity >= min_similarity:
                        formatted_results.append({
                            'content': doc,
                            'metadata': metadata or {},
                            'similarity_score': round(similarity, 4)
                        })
                
                logger.info(f"Retrieved {len(formatted_results)} relevant documents")
            else:
                logger.warning("No results returned from ChromaDB")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def format_context_for_prompt(self, context_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string for the LLM.
        
        Args:
            context_docs: List of retrieved documents
        
        Returns:
            Formatted context string
        """
        if not context_docs:
            return "No relevant medical information found in the database."
        
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            content = doc['content']
            similarity = doc.get('similarity_score', 0)
            source = doc.get('metadata', {}).get('source', 'Unknown')
            
            context_parts.append(
                f"[Source {i} - Relevance: {similarity:.2%}]\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_rag_context(
        self,
        query: str,
        n_results: int = 3
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get formatted context and source documents for a query.
        
        Args:
            query: User's question
            n_results: Number of documents to retrieve
        
        Returns:
            Tuple of (formatted_context, source_documents)
        """
        # Retrieve relevant documents
        sources = self.retrieve_context(query, n_results=n_results)
        
        # Format for prompt
        context = self.format_context_for_prompt(sources)
        
        return context, sources
    
    def is_healthy(self) -> bool:
        """Check if ChromaDB connection is healthy."""
        try:
            if self.collection:
                self.collection.count()
                return True
            return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Singleton instance
_rag_engine_instance = None


def get_rag_engine() -> RAGEngine:
    """Get or create RAG engine singleton."""
    global _rag_engine_instance
    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine()
    return _rag_engine_instance
