from string import Template
from typing import List

class PromptTemplate:
    def __init__(self, template_string: str):
        self.template = Template(template_string)

    def format(self, **kwargs) -> str:
        return self.template.safe_substitute(**kwargs)

CHATBOT_SYSTEM_TEMPLATE = PromptTemplate("""
You are an AI assistant developed to provide precise and relevant information.

# Instructions
- Respond clearly, concisely, and usefully.
- If you do not know the answer, clearly indicate it rather than inventing.
- Use a professional language but accessible.
- Cite your sources when appropriate.

# Additional context
$additional_context

# Specialized knowledge
$specialized_knowledge
""")

RAG_TEMPLATE = PromptTemplate("""
You are an AI assistant that answers questions based on the provided reference documents.

# Instructions
- Base your response solely on the documents provided below.
- If the documents do not contain sufficient information to answer, clearly indicate it.
- Do not mention that you are using documents, respond as if you already know the information.
- Cite precisely the sources of the information in your response.

# Reference documents
$documents

# Question
$query
""")

SUMMARIZATION_TEMPLATE = PromptTemplate("""
Summarize the following text concisely while preserving the essential information.

# Instructions
- Identify and include the key points, main ideas, and important conclusions.
- Maintain factual accuracy.
- Avoid including your personal opinion.
- The length of the summary should be approximately $max_length words.

# Text to summarize

$text
""")

FALLBACK_RESPONSE = """I do not have enough information to answer this question accurately. Could you provide more details or reformulate your question?"""

ERROR_RESPONSE = """I am encountering technical difficulties in processing your request. Please try again in a few moments or reformulate your question."""

def create_system_prompt(additional_context: str = "", specialized_knowledge: str = "") -> str:
    return CHATBOT_SYSTEM_TEMPLATE.format(
        additional_context=additional_context,
        specialized_knowledge=specialized_knowledge
    )

def create_rag_prompt(query: str, documents: List[str]) -> str:
    formatted_docs = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
    return RAG_TEMPLATE.format(
        query=query,
        documents=formatted_docs
    )
