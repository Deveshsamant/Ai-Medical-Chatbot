"""
FastAPI Backend for Medical Chatbot with RAG.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import time
from typing import Dict

from models import ChatRequest, ChatResponse, HealthResponse, ErrorResponse, Source
from rag_engine import get_rag_engine
from llm_client import get_llm_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
rag_engine = None
llm_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global rag_engine, llm_client
    logger.info("Initializing Medical Chatbot Backend...")
    
    try:
        rag_engine = get_rag_engine()
        logger.info("✓ RAG Engine initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize RAG Engine: {e}")
        raise
    
    try:
        llm_client = get_llm_client()
        logger.info("✓ LLM Client initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize LLM Client: {e}")
        raise
    
    logger.info("Backend initialization complete!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Medical Chatbot Backend...")


# Create FastAPI app
app = FastAPI(
    title="Medical Chatbot API",
    description="RAG-powered medical chatbot with ChromaDB and LLM integration",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Medical Chatbot API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns the status of ChromaDB and LLM connections.
    """
    chromadb_healthy = rag_engine.is_healthy() if rag_engine else False
    llm_available = llm_client.is_available() if llm_client else False
    
    status = "healthy" if (chromadb_healthy and llm_available) else "degraded"
    
    return HealthResponse(
        status=status,
        chromadb_connected=chromadb_healthy,
        llm_available=llm_available
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat endpoint for medical queries.
    
    Processes user messages using RAG and LLM to generate responses.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing chat request: '{request.message[:50]}...'")
        
        # Step 1: Retrieve relevant context from ChromaDB
        context, sources = rag_engine.get_rag_context(
            query=request.message,
            n_results=request.max_sources
        )
        
        logger.info(f"Retrieved {len(sources)} source documents")
        
        # Step 2: Generate response using LLM
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.conversation_history
        ] if request.conversation_history else []
        
        response_text = llm_client.generate_response(
            user_question=request.message,
            context=context,
            conversation_history=conversation_history,
            temperature=request.temperature
        )
        
        logger.info("Response generated successfully")
        
        # Step 3: Format response
        processing_time = time.time() - start_time
        
        # Calculate confidence based on source relevance
        confidence = None
        if sources:
            avg_similarity = sum(s.get('similarity_score', 0) for s in sources) / len(sources)
            confidence = round(avg_similarity, 2)
        
        # Format sources
        formatted_sources = [
            Source(
                content=s['content'][:200] + "..." if len(s['content']) > 200 else s['content'],
                metadata=s.get('metadata', {}),
                similarity_score=s.get('similarity_score')
            )
            for s in sources
        ]
        
        return ChatResponse(
            message=response_text,
            sources=formatted_sources,
            confidence=confidence,
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )


@app.get("/stats", tags=["Stats"])
async def get_stats():
    """Get database statistics."""
    try:
        if not rag_engine or not rag_engine.collection:
            raise HTTPException(status_code=503, detail="ChromaDB not available")
        
        doc_count = rag_engine.collection.count()
        
        return {
            "total_documents": doc_count,
            "collection_name": rag_engine.collection.name,
            "embedding_model": rag_engine.embedding_model
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
