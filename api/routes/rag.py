import ollama
from fastapi import APIRouter, Depends, HTTPException
from api.schemas import (
    RAGRequest, RAGResponse, SourceChunk,
    SummarizeRequest, SummarizeResponse,
    ExtractRequest, ExtractResponse
)
from api.middleware.auth import verify_api_key
from config import MODEL

router = APIRouter(prefix="/rag", tags=["RAG & Knowledge"])


@router.post(
    "/ask",
    response_model=RAGResponse,
    summary="Ask a question using RAG",
    description="Ask a question and get an answer grounded in your document knowledge base."
)
async def ask_rag(
    request: RAGRequest,
    api_key: str = Depends(verify_api_key)
):
    """RAG endpoint — retrieves relevant docs and generates grounded answers."""
    try:
        from rag.knowledge_base import search, answer

        chunks = search(
            query=request.question,
            top_k=request.top_k,
            source_filter=request.source_filter,
            level_filter=request.level_filter
        )

        if not chunks:
            return RAGResponse(
                answer="No relevant information found in the knowledge base.",
                sources=[],
                question=request.question,
                chunks_retrieved=0
            )

        context = "\n\n---\n\n".join(
            f"[Source: {c['source']} | Score: {c['score']}]\n{c['text']}"
            for c in chunks
        )

        prompt = f"""Answer the question using ONLY the context below.
If the answer is not present, say you do not have that information.
Always mention which source you used.

CONTEXT:
{context}

QUESTION: {request.question}

ANSWER:"""

        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.3}
        )

        answer_text = response["message"]["content"]

        sources = []
        if request.show_sources:
            sources = [
                SourceChunk(
                    text=c["text"][:200],
                    source=c["source"],
                    score=c["score"]
                )
                for c in chunks
            ]

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            question=request.question,
            chunks_retrieved=len(chunks)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")


@router.get(
    "/documents",
    summary="List indexed documents",
    description="Returns all documents currently indexed in the knowledge base."
)
async def list_documents(api_key: str = Depends(verify_api_key)):
    """List all documents in the knowledge base."""
    try:
        from rag.knowledge_base import list_documents as kb_list
        docs = kb_list()
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize text",
    description="Summarize any text using zero-shot, few-shot, or chain-of-thought prompting."
)
async def summarize(
    request: SummarizeRequest,
    api_key: str = Depends(verify_api_key)
):
    """Summarize text using the chosen prompting strategy."""
    try:
        from summarizer import summarize as do_summarize
        result = do_summarize(request.text, mode=request.style.value)

        return SummarizeResponse(
            summary=result if result else "Summarization failed.",
            style=request.style.value,
            original_length=len(request.text),
            summary_length=len(result) if result else 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarize error: {str(e)}")


@router.post(
    "/extract",
    response_model=ExtractResponse,
    summary="Extract structured data",
    description="Extract named entities, contacts, meeting notes, or review data from text."
)
async def extract(
    request: ExtractRequest,
    api_key: str = Depends(verify_api_key)
):
    """Extract structured data from text using Pydantic-validated schemas."""
    try:
        from extractor import (
            extract_entities, extract_contact,
            extract_meeting, extract_review
        )

        extractors = {
            "entities": extract_entities,
            "contact": extract_contact,
            "meeting": extract_meeting,
            "review": extract_review
        }

        fn = extractors.get(request.extract_type, extract_entities)
        result = fn(request.text)

        if result:
            result_dict = result.model_dump()
            return ExtractResponse(
                result=result_dict,
                extract_type=request.extract_type,
                success=True
            )
        else:
            return ExtractResponse(
                result={},
                extract_type=request.extract_type,
                success=False
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extract error: {str(e)}")