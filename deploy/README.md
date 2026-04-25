# Deployment Guide

## Local Development (without Docker)
uvicorn api.main:app --reload --port 8000
streamlit run frontend/app.py --server.port 8501

## Docker Compose (recommended)

### Start all services
docker compose up --build

### Start in background
docker compose up -d --build

### Stop all services
docker compose down

### View logs
docker compose logs -f api
docker compose logs -f frontend
docker compose logs -f ollama

### Pull Ollama model inside container
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text

## Services and Ports
- Ollama LLM:          http://localhost:11434
- FastAPI Backend:     http://localhost:8000
- API Docs (Swagger):  http://localhost:8000/docs
- Streamlit Frontend:  http://localhost:8501
- Streaming API:       http://localhost:8001

## Environment Variables
Copy deploy/env.example to .env and fill in your values:
cp deploy/env.example .env

## Production Deployment (Railway)
1. Push to GitHub
2. Connect repo to Railway (railway.app)
3. Add environment variables
4. Deploy — Railway auto-detects Docker

## Production Deployment (Render)
1. Create new Web Service on render.com
2. Connect GitHub repo
3. Set Build Command: docker build
4. Set Start Command: uvicorn api.main:app --host 0.0.0.0 --port 8000
5. Add environment variables
6. Deploy