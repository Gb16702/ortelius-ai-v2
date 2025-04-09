from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import field_validator

class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="List of messages of the conversation")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature of the model")
    max_tokens: int = Field(default=1000, gt=0, description="Maximum number of tokens in the response")
    is_rag_enabled: bool = Field(default=False, description="Whether RAG is enabled")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")

        for i, message in enumerate(v):
            if not isinstance(message, dict):
                raise ValueError(f"Message {i} must be a dictionary")

            if "role" not in message:
                raise ValueError(f"Message {i} must contain 'role' key")

            if "content" not in message:
                raise ValueError(f"Message {i} must contain 'content' key")

            if not message.get("content"):
                raise ValueError("Message content cannot be empty")

        return v




class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class ChatCompletionResponse(BaseModel):
    message: ChatMessage
    usage: Optional[Dict[str, Any]] = None
