import logfire

from typing import AsyncGenerator, List, Dict, Any
from openai import AsyncOpenAI
from pydantic_ai import Agent
from fastapi import HTTPException, status

from app.core.config import settings
from app.utils.prompt_templates import create_system_prompt, ERROR_RESPONSE
from app.models.chat import ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from app.services.language_service import LanguageService
from pydantic import ValidationError

class ChatService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.agent = Agent(f"openai:{settings.CHAT_MODEL}")

    def _extract_user_prompt(self, messages: List[ChatMessage]) -> ChatMessage:
        return next((message for message in reversed(messages) if message.role == "user"), None)

    def _ensure_system_message(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        detected_language, _ = LanguageService.detect_language(self._extract_user_prompt(messages).content)
        system_prompt = create_system_prompt()

        if detected_language != "en":
            instructions = f"The user is speaking in {detected_language}. ALWAYS respond in the same language as the user."
            system_prompt = f"{instructions}\n\n{system_prompt}"

        if not any(message.role == "system" for message in messages):
            system_message = ChatMessage(role="system", content=system_prompt)
            return [system_message] + messages

        return messages

    async def get_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            messages = self._ensure_system_message(request.messages)

            user_prompt = self._extract_user_prompt(messages).content
            system_message = next((message.content for message in messages if message.role == "system"), '')

            model_settings = {
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }

            temp_agent = Agent(
                f"openai:{settings.CHAT_MODEL}",
                system_prompt=system_message
            )

            response = await temp_agent.run(
                user_prompt=user_prompt,
                model_settings=model_settings,
            )

            logfire.info("Chat completion response",
                        response=response.data,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens)

            return ChatCompletionResponse(
                message=ChatMessage(role="assistant", content=response.data),
                usage=None
            )

        except ValidationError as e:
            logfire.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=e.errors()
            )

        except HTTPException:
            raise
        except Exception as e:
            logfire.error(f"Error in chat completion: {str(e)}")

            return ChatCompletionResponse(
                message=ChatMessage(role="assistant", content=ERROR_RESPONSE),
                usage=None
            )