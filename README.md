# 🧠 My AI Engineer Journey

> Building a production-grade AI assistant from scratch — one feature at a time.

## Stack
Python 3.11 · Ollama (local LLM) · HuggingFace · ChromaDB · LangChain · FastAPI · Docker

## Setup
pip install -r requirements.txt
ollama pull llama3.2
python chat.py

## Run with a persona
python chat.py --persona mentor
python chat.py --persona socratic
python chat.py --persona pirate

## Progress
- [x] Multi-turn CLI chatbot with Ollama, personas, memory and session logging