from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, Any]]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)
    is_rag_enabled: bool = Field(default=False)

class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class ChatCompletionResponse(BaseModel):
    message: ChatMessage
    usage: Optional[Dict[str, Any]] = None
