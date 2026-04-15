import asyncio
import json
import ollama
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import MODEL, TEMPERATURE

app = FastAPI(
    title="Streaming AI Endpoints",
    description="Token-by-token streaming responses over HTTP using SSE"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StreamRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful AI assistant."
    persona: str = "default"

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Explain machine learning in simple terms",
                "persona": "default"
            }
        }


PERSONAS = {
    "default":  "You are a helpful AI assistant for an AIML student.",
    "mentor":   "You are a senior ML engineer mentoring a student.",
    "socratic": "You are a Socratic teacher who guides through questions.",
    "pirate":   "You are a pirate who became an ML expert. Arrr."
}


async def generate_stream(message: str, system_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]

    try:
        for chunk in ollama.chat(
            model=MODEL,
            messages=messages,
            stream=True,
            options={"temperature": TEMPERATURE}
        ):
            token = chunk["message"]["content"]
            event = f"data: {json.dumps({'token': token})}\n\n"
            yield event
            await asyncio.sleep(0)

        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/stream/chat")
async def stream_chat(request: StreamRequest):
    system_prompt = PERSONAS.get(request.persona, PERSONAS["default"])

    return StreamingResponse(
        generate_stream(request.message, system_prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/stream/rag")
async def stream_rag(question: str, top_k: int = 3):
    from rag.knowledge_base import search

    async def rag_stream():
        chunks = search(query=question, top_k=top_k)

        if not chunks:
            yield f"data: {json.dumps({'token': 'No relevant information found.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        sources = [{"source": c["source"], "score": c["score"]} for c in chunks]
        yield f"data: {json.dumps({'sources': sources, 'type': 'sources'})}\n\n"

        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}"
            for c in chunks
        )

        prompt = f"""Answer using ONLY the context. Cite your source.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        for chunk in ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": 0.3}
        ):
            token = chunk["message"]["content"]
            yield f"data: {json.dumps({'token': token, 'type': 'token'})}\n\n"
            await asyncio.sleep(0)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        rag_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL, "streaming": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "streaming.stream_endpoint:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )