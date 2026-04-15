import time
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.routes.chat import router as chat_router
from api.routes.rag import router as rag_router
from api.middleware.auth import verify_api_key
from api.schemas import HealthResponse
from config import MODEL

# ── APP SETUP ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="My AI Engineer Journey — API",
    description="""
A production-grade AI API exposing all features built over 30 days.

## Features
- **Chat** — Multi-turn conversation with personas and memory
- **RAG** — Document Q&A grounded in your knowledge base
- **Summarize** — Zero-shot, few-shot, chain-of-thought summarization
- **Extract** — Named entities, contacts, meetings, and reviews

## Authentication
All endpoints require an API key in the `X-API-Key` header.
Dev key: `dev-key-aiml-journey-2024`

## Docs
- Interactive docs: `/docs`
- OpenAPI schema: `/openapi.json`
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # in production: specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REQUEST TIMING MIDDLEWARE ──────────────────────────────────────────────
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Response-Time header to every response."""
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 2)
    response.headers["X-Response-Time"] = f"{elapsed}ms"
    return response

# ── ROUTERS ───────────────────────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(rag_router)

# ── CORE ENDPOINTS ────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Check if the API is running and which features are available."
)
async def health():
    """Health check endpoint — no auth required."""
    return HealthResponse(
        status="healthy",
        model=MODEL,
        version="1.0.0",
        features=["chat", "rag", "summarize", "extract", "memory"]
    )


@app.get(
    "/",
    tags=["System"],
    summary="API root"
)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "My AI Engineer Journey API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "POST /chat/",
            "rag": "POST /rag/ask",
            "documents": "GET /rag/documents",
            "summarize": "POST /rag/summarize",
            "extract": "POST /rag/extract"
        }
    }


@app.get(
    "/models",
    tags=["System"],
    summary="List available models"
)
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available Ollama models."""
    import ollama as ol
    try:
        models = ol.list()
        return {
            "models": [m["name"] for m in models.get("models", [])],
            "current": MODEL
        }
    except Exception as e:
        return {"models": [], "error": str(e), "current": MODEL}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )