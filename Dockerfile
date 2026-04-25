# ── Base image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# ── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────────────────────────────────────────
# Copy requirements first for layer caching
# If requirements.txt hasn't changed, Docker reuses the cached layer
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn \
        ollama \
        python-dotenv \
        chromadb \
        pydantic \
        pypdf \
        streamlit \
        requests \
        langchain \
        langchain-community \
        langchain-ollama \
        langchain-core \
        langchain-text-splitters \
        rank-bm25 \
        rouge-score \
        networkx

# ── Application code ─────────────────────────────────────────────────────────
COPY . .

# ── Create necessary directories ──────────────────────────────────────────────
RUN mkdir -p \
    chroma_db \
    data/docs \
    data/images \
    chat_logs \
    agent_logs \
    guardrails/logs \
    observability/traces \
    observability/metrics \
    observability/alerts \
    eval/reports \
    finetuning/data

# ── Environment variables ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL=llama3.2

# ── Expose ports ──────────────────────────────────────────────────────────────
# 8000 = FastAPI backend
# 8001 = Streaming endpoint
# 8501 = Streamlit frontend
EXPOSE 8000 8001 8501

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Default command ────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]