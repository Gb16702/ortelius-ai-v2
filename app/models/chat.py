from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import field_validator
from pydantic_core import PydanticCustomError

class ChatMessage(BaseModel):
    class Config:
        extra = "forbid"
        validate_assignment = True
        frozen = True

    role: str = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="The creation date of the message")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        v = v.strip()

        if not v:
            raise PydanticCustomError("content_empty", "Content cannot be empty")
        if len(v) > 1000:
            raise PydanticCustomError("content_too_long", "Content cannot be longer than 1000 characters")
        return v

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed_roles = {"user", "assistant", "system"}
        if v not in allowed_roles:
            raise PydanticCustomError(
                "invalid_role",
                f"Role must be one of {allowed_roles}, got '{v}'"
            )
        return v

    @field_validator("created_at")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise PydanticCustomError(
                "invalid_timestamp",
                "created_at must be a valid ISO format timestamp"
            )
        return v

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of messages of the conversation")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature of the model")
    max_tokens: int = Field(default=1000, gt=0, description="Maximum number of tokens in the response")
    is_rag_enabled: bool = Field(default=False, description="Whether RAG is enabled")

class ChatCompletionResponse(BaseModel):
    message: ChatMessage
    usage: Optional[Dict[str, Any]] = None
