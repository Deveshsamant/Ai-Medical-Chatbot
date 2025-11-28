"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's message")
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=[],
        description="Previous conversation messages for context"
    )
    max_sources: Optional[int] = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of source documents to retrieve"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature for response generation"
    )


class Source(BaseModel):
    """Source document information."""
    content: str = Field(..., description="Relevant text from source")
    metadata: Dict[str, Any] = Field(default={}, description="Source metadata")
    similarity_score: Optional[float] = Field(None, description="Similarity score")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    message: str = Field(..., description="Assistant's response")
    sources: List[Source] = Field(default=[], description="Source documents used")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Response confidence")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    chromadb_connected: bool = Field(..., description="ChromaDB connection status")
    llm_available: bool = Field(..., description="LLM availability status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now)
