import logfire

from typing import AsyncGenerator, List, Dict, Any
from openai import AsyncOpenAI
from pydantic_ai import Agent

from app.core.config import settings
from app.utils.prompt_templates import create_system_prompt, FALLBACK_RESPONSE
from app.models.chat import ChatCompletionRequest, ChatCompletionResponse

class ChatService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.agent = Agent(f"openai:{settings.CHAT_MODEL}")

    def _ensure_system_message(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not any(message.get("role") == "system" for message in messages):
            messages.insert(0, {"role": "system", "content": create_system_prompt()})
        return messages

    async def get_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            messages = self._ensure_system_message(request.messages)

            response = await self.agent.run(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            logfire.info("Chat completion response",
                        response=response.data,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens)

            return ChatCompletionResponse(
                message={ "role": "assistant", "content": response.data },
                usage=None
            )

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

            async for chunk in self.agent.stream(messages=messages, temperature=request.temperature, max_tokens=request.max_tokens):
                if chunk and hasattr(chunk, "data"):
                    yield chunk.data

        except Exception as e:
            logfire.error(f"Error in streaming chat completion: {str(e)}")
            yield FALLBACK_RESPONSE
