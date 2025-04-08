from typing import List, Dict, Optional

from app.utils.prompt_templates import (
    create_system_prompt,
    create_rag_prompt,
    SUMMARIZATION_TEMPLATE
)

class PromptService:

    @staticmethod
    def create_chat_prompt(
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        additional_context: str = "",
        specialized_knowledge: str = ""
    ) -> List[Dict[str, str]]:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            default_system = create_system_prompt(
                additional_context=additional_context,
                specialized_knowledge=specialized_knowledge
            )
            messages.append({"role": "system", "content": default_system})

        if chat_history:
            messages.extend(chat_history)

        messages.append({"role": "user", "content": query})

        return messages

    @staticmethod
    def create_rag_prompt_with_context(
        query: str,
        contexts: List[str],
        include_chain_of_thought: bool = True
    ) -> str:
        base_prompt = create_rag_prompt(query=query, documents=contexts)

        if include_chain_of_thought:
            base_prompt += "\n\nThink step by step to provide an accurate response by relying on the documents provided."

        return base_prompt

    @staticmethod
    def create_summarization_prompt(text: str, max_length: int = 150) -> str:
        return SUMMARIZATION_TEMPLATE.format(
            text=text,
            max_length=max_length
        )

    @staticmethod
    def enhance_query_for_retrieval(query: str) -> str:
        enhanced_query = f"I am looking for information about: {query}"
        enhanced_query += "\nComplete and detailed response containing all important aspects of this subject:"

        return enhanced_query