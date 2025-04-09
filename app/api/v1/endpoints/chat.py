import json

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Body
from fastapi.responses import StreamingResponse

from app.services.chat_service import ChatService
from app.services.prompt_service import PromptService
from app.models.chat import ChatCompletionRequest

router = APIRouter(tags=["chat"])

@router.post("/chat")
async def chat(request: ChatCompletionRequest):
        chat_service = ChatService()
        return await chat_service.get_chat_completion(request)
