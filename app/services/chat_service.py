import logfire

from typing import AsyncGenerator, List, Dict, Any
from openai import AsyncOpenAI
from pydantic_ai import Agent
from fastapi import HTTPException, status

from app.core.config import settings
from app.utils.prompt_templates import create_system_prompt, FALLBACK_RESPONSE
from app.models.chat import ChatCompletionRequest, ChatCompletionResponse
from app.services.language_service import LanguageService
from pydantic import ValidationError

class ChatService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.agent = Agent(f"openai:{settings.CHAT_MODEL}")

    def _ensure_system_message(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        user_prompt = next((message.get("content", '') for message in reversed(messages) if message.get("role") == "user"), '')
        if not user_prompt or not user_prompt.strip():
            ERROR_MESSAGE = "The prompt cannot be empty"
            logfire.error(ERROR_MESSAGE)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGE
            )

        detected_language, _ = LanguageService.detect_language(user_prompt)
        system_prompt = create_system_prompt()

        if detected_language != "en":
            instructions = f"The user is speaking in {detected_language}. ALWAYS respond in the same language as the user."
            system_prompt = f"{instructions}\n\n{system_prompt}"

        if not any(message.get("role") == "system" for message in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})

        return messages

    async def get_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            messages = self._ensure_system_message(request.messages)

            user_prompt = next((message.get("content", '') for message in reversed(messages) if message.get("role") == "user"), '')
            system_message = next((message.get("content", '') for message in messages if message.get("role") == "system"), '')

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
                message={ "role": "assistant", "content": response.data },
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
                message={ "role": "assistant", "content": FALLBACK_RESPONSE },
                usage=None
            )

    async def _get_rag_context(self, input: str) -> str | None:
        try:
            logfire.info("Getting RAG context", input=input)

            embedding_response = await self.client.embeddings.create(
                input=input,
                model=settings.EMBEDDING_MODEL
            )

            embedding = embedding_response.data[0].embedding

            return None

        except Exception as e:
            logfire.error(f"Error in getting RAG context: {str(e)}")
            return None

    async def get_streaming_chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        try:
            messages = self._ensure_system_message(request.messages)

            if request.is_rag_enabled:
                user_query = next((message.get("content", '') for message in reversed(request.messages) if message.get("role") == "user"), '')

                ctx = await self._get_rag_context(user_query)
                if ctx:
                    system_index = next((i for i, message in enumerate(messages) if message.get("role") == "system"), 0)
                    messages.insert(system_index + 1, {
                        "role": "system",
                        "content": f"Context{ctx}"
                    })

            user_prompt = next((message.get("content", '') for message in reversed(messages) if message.get("role") == "user"), '')
            system_message = next((message.get("content", '') for message in messages if message.get("role") == "system"), '')

            model_settings = {
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }

            temp_agent = Agent(
                f"openai:{settings.CHAT_MODEL}",
                system_prompt=system_message
            )

            async with temp_agent.run_stream(
                user_prompt=user_prompt,
                model_settings=model_settings,
            ) as streamed_response:
                async for text in streamed_response.stream_text():
                    yield text

        except HTTPException:
            raise
        except Exception as e:
            logfire.error(f"Error in streaming chat completion: {str(e)}")
            yield FALLBACK_RESPONSE
