from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class PersonaEnum(str, Enum):
    default = "default"
    mentor = "mentor"
    socratic = "socratic"
    pirate = "pirate"


class StyleEnum(str, Enum):
    zero = "zero"
    few = "few"
    cot = "cot"


# ── CHAT SCHEMAS ──────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: user or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's message"
    )
    persona: PersonaEnum = Field(
        default=PersonaEnum.default,
        description="Assistant persona to use"
    )
    history: list[ChatMessage] = Field(
        default=[],
        description="Previous conversation messages"
    )
    user_id: str = Field(
        default="api_user",
        description="User ID for memory isolation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is machine learning?",
                "persona": "default",
                "history": [],
                "user_id": "roshini"
            }
        }


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Assistant's response")
    persona: str = Field(..., description="Persona used")
    user_id: str = Field(..., description="User ID")
    tokens_estimated: int = Field(..., description="Estimated token count")


# ── RAG SCHEMAS ────────────────────────────────────────────────────────────

class RAGRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Question to answer from documents"
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of chunks to retrieve"
    )
    source_filter: Optional[str] = Field(
        default=None,
        description="Filter by document filename"
    )
    level_filter: Optional[str] = Field(
        default=None,
        description="Filter by difficulty level"
    )
    show_sources: bool = Field(
        default=True,
        description="Include source attribution in response"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is RAG and how does it work?",
                "top_k": 3,
                "show_sources": True
            }
        }


class SourceChunk(BaseModel):
    text: str
    source: str
    score: float


class RAGResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: list[SourceChunk] = Field(
        default=[],
        description="Source chunks used to generate the answer"
    )
    question: str = Field(..., description="Original question")
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved")


# ── SUMMARIZE SCHEMAS ──────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=50,
        max_length=10000,
        description="Text to summarize"
    )
    style: StyleEnum = Field(
        default=StyleEnum.zero,
        description="Prompting strategy to use"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Machine learning is transforming every industry...",
                "style": "zero"
            }
        }


class SummarizeResponse(BaseModel):
    summary: str
    style: str
    original_length: int
    summary_length: int


# ── EXTRACT SCHEMAS ────────────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Text to extract information from"
    )
    extract_type: str = Field(
        default="entities",
        description="Type: entities, contact, meeting, review"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Apple CEO Tim Cook announced a new facility in Bangalore...",
                "extract_type": "entities"
            }
        }


class ExtractResponse(BaseModel):
    result: dict = Field(..., description="Extracted structured data")
    extract_type: str
    success: bool


# ── HEALTH SCHEMAS ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    features: list[str]