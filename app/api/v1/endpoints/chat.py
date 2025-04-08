import json

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Body
from fastapi.responses import StreamingResponse

from app.services.chat_service import ChatService
from app.services.prompt_service import PromptService
from app.models.chat import ChatCompletionRequest

router = APIRouter(tags=["chat"])

@router.post("/chat")
async def chat(request: Dict[str, Any] = Body(...)):
    try:
        messages = request.get("messages", [])
        temperature = request.get("temperature", 0.2)
        max_tokens = request.get("max_tokens", 1000)
        is_rag_enabled = request.get("is_rag_enabled", False)

        chat_service = ChatService()

        chat_request = ChatCompletionRequest(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            is_rag_enabled=is_rag_enabled
        )


        return await chat_service.get_chat_completion(chat_request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while processing the request: {str(e)}"
        )

@router.post("/chat/stream")
async def chat_stream(request: Dict[str, Any] = Body(...)):
    try:
        messages = request.get("messages", [])
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 1000)
        is_rag_enabled = request.get("is_rag_enabled", False)


        chat_service = ChatService()

        chat_request = ChatCompletionRequest(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            is_rag_enabled=is_rag_enabled
        )

        async def generate():
            try:
                async for chunk in chat_service.get_streaming_chat_completion(chat_request):
                    data = {"chunk": chunk}
                    yield f"data: {json.dumps(data)}\n\n"

                data = {"chunk": '', "finish_reason": "stop"}
                yield f"data: {json.dumps(data)}\n\n"
            except Exception as e:
                data = {"error": str(e)}
                yield f"data: {json.dumps(data)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while initializing streaming: {str(e)}"
        )