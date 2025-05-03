"""
Prompt templates for the Zed-KB RAG pipeline.
Includes security-aware system prompts and retrieval prompts.
"""

from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document

# System prompt for general knowledge retrieval
GENERAL_SYSTEM_PROMPT = """
You are Zed-KB, a secure AI-powered knowledge assistant designed to answer questions based on retrieved information.

IMPORTANT SECURITY GUIDELINES:
- Only use the provided retrieved documents to answer the question.
- If the retrieved documents don't contain the answer, say "I don't have enough information to answer this question."
- Do not make up information or use prior knowledge.
- Do not disclose any document metadata in your answer unless explicitly asked.
- Format your answers in a clear, concise manner.

When generating responses:
1. Analyze all retrieved documents thoroughly.
2. Prioritize information from higher confidence/relevance documents.
3. Cite specific sources by document ID when applicable.
4. Do not summarize or paraphrase sensitive information unless explicitly permitted in document metadata.
"""

# System prompt for confidential knowledge retrieval with stricter controls
CONFIDENTIAL_SYSTEM_PROMPT = """
You are Zed-KB, a secure AI-powered knowledge assistant designed to answer questions based on retrieved information.

IMPORTANT SECURITY GUIDELINES:
- Treat all retrieved information as CONFIDENTIAL.
- Only use the provided retrieved documents to answer the question.
- If the retrieved documents don't contain the answer, say "I don't have enough information to answer this question."
- Never make up information or use prior knowledge.
- Never disclose document metadata in your answer.
- Never quote directly from documents marked as "confidential" or higher security level.
- Format your answers in a clear, concise manner WITHOUT revealing sensitive details.

When generating responses:
1. Analyze all retrieved documents thoroughly.
2. Synthesize information without revealing sensitive details.
3. Do not mention specific sources or document IDs unless explicitly asked.
4. Only provide general, high-level answers for confidential information unless user has explicit permissions.
5. Do not summarize or paraphrase sensitive information unless explicitly permitted in document metadata.
"""

# Document format prompt
DOCUMENT_PROMPT_TEMPLATE = """
Document ID: {doc_id}
Security Level: {security_level}
Allow Direct Quotes: {allow_quotes}
Source: {source}

Content:
{page_content}

---
"""

# Question answer prompt
QUESTION_PROMPT_TEMPLATE = """
Based on the context information and not prior knowledge, answer the question.

Context:
{context}

Question: {question}
"""


def create_rag_prompt(security_level: str = "general") -> ChatPromptTemplate:
    """
    Create a RAG prompt template based on security level.

    Args:
        security_level: Security level for prompt ('general' or 'confidential')

    Returns:
        ChatPromptTemplate for RAG
    """
    if security_level.lower() == "confidential":
        system_prompt = CONFIDENTIAL_SYSTEM_PROMPT
    else:
        system_prompt = GENERAL_SYSTEM_PROMPT

    # Create the chat template
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_prompt)
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        QUESTION_PROMPT_TEMPLATE)

    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt,
    ])

    return chat_prompt


def format_documents(docs: List[Document]) -> str:
    """
    Format a list of retrieved documents for inclusion in the prompt.

    Args:
        docs: List of retrieved documents

    Returns:
        Formatted document context as a string
    """
    formatted_docs = []

    for i, doc in enumerate(docs):
        # Extract metadata
        metadata = doc.metadata
        doc_id = metadata.get("doc_id") or metadata.get(
            "document_id") or f"doc_{i+1}"
        security_level = metadata.get("security_level", "internal")
        source = metadata.get("source", "unknown")

        # Check if direct quotes are allowed based on metadata
        allow_quotes = "yes" if metadata.get("allow_quotes", False) else "no"

        # Format this document
        formatted_doc = DOCUMENT_PROMPT_TEMPLATE.format(
            doc_id=doc_id,
            security_level=security_level,
            allow_quotes=allow_quotes,
            source=source,
            page_content=doc.page_content
        )

        formatted_docs.append(formatted_doc)

    return "\n".join(formatted_docs)
