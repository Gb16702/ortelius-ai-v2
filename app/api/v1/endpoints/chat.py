from fastapi import APIRouter

from app.services.chat_service import ChatService
from app.models.chat import ChatCompletionRequest

router = APIRouter(tags=["chat"])

@router.post("/chat")
async def chat(request: ChatCompletionRequest):
        chat_service = ChatService()
        return await chat_service.get_chat_completion(request)
