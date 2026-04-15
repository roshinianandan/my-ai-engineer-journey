import ollama
from fastapi import APIRouter, Depends, HTTPException
from api.schemas import ChatRequest, ChatResponse
from api.middleware.auth import verify_api_key
from config import MODEL, TEMPERATURE, MAX_TOKENS

router = APIRouter(prefix="/chat", tags=["Chat"])

PERSONAS = {
    "default":  "You are a helpful AI assistant for an AIML student. Be concise and use examples.",
    "mentor":   "You are a senior ML engineer mentoring a student. Be encouraging and technical.",
    "socratic": "You are a Socratic teacher. Guide through questions, never give direct answers.",
    "pirate":   "You are a pirate who became an ML expert. Use nautical metaphors. Arrr."
}


@router.post(
    "/",
    response_model=ChatResponse,
    summary="Send a chat message",
    description="Send a message and get a response from the AI assistant. Supports personas and conversation history."
)
async def chat(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """Chat endpoint — the core of the AI assistant."""
    try:
        system_prompt = PERSONAS.get(request.persona.value, PERSONAS["default"])

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in request.history[-10:]:  # limit to last 10 messages
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Add current message
        messages.append({"role": "user", "content": request.message})

        response = ollama.chat(
            model=MODEL,
            messages=messages,
            stream=False,
            options={
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS
            }
        )

        reply = response["message"]["content"]
        estimated_tokens = len(request.message.split()) + len(reply.split())

        return ChatResponse(
            reply=reply,
            persona=request.persona.value,
            user_id=request.user_id,
            tokens_estimated=estimated_tokens
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.get(
    "/personas",
    summary="List available personas",
    description="Returns all available assistant personas."
)
async def list_personas(api_key: str = Depends(verify_api_key)):
    """List all available chat personas."""
    return {
        "personas": list(PERSONAS.keys()),
        "descriptions": {
            k: v[:80] + "..." for k, v in PERSONAS.items()
        }
    }