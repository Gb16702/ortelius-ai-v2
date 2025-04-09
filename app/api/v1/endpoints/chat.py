from fastapi import APIRouter

from app.services.chat_service import ChatService
from app.models.message import CompletionRequest

router = APIRouter(tags=["chat"])

@router.post("/chat")
async def chat(request: CompletionRequest):
        chat_service = ChatService()
        return await chat_service.get_chat_completion(request)
