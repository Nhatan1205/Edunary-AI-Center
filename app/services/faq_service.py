"""
FAQ service — handles FAQ queries using RAG (Retrieval-Augmented Generation).
Extracted from intent_handlers.py: handle_faq() and query_data.py: query_rag().
"""

import time

from langchain_core.prompts import ChatPromptTemplate

from app.prompts.faq_prompts import FAQ_RAG_PROMPT
from app.llm.factory import create_llm
from app.store.vector_store import similarity_search
from app.core.logging import get_logger

logger = get_logger(__name__)


def query_rag(query_text: str) -> str:
    """
    Query the RAG pipeline: search vector store → build context → ask LLM.

    Args:
        query_text: The user's FAQ question.

    Returns:
        str: The LLM-generated answer based on retrieved context.
    """
    # Search vector store
    # _t0 = time.perf_counter()
    results = similarity_search(query_text, k=5)
    # logger.info(f"similarity_search took {time.perf_counter() - _t0:.3f}s")

    # Combine all chunks into context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Build prompt and invoke LLM
    prompt_template = ChatPromptTemplate.from_template(FAQ_RAG_PROMPT)
    prompt = prompt_template.format(context=context_text, question=query_text)

    llm = create_llm()
    response_result = llm.invoke(prompt)

    # Handle both string (OllamaLLM) and AIMessage (ChatOpenAI, ChatGoogleGenerativeAI)
    response_text = response_result.content if hasattr(response_result, "content") else response_result

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    logger.info(f"FAQ answered with {len(sources)} sources")

    return response_text


def handle_faq(message: str) -> dict:
    """
    Handle an FAQ intent.

    Args:
        message: The user's FAQ question.

    Returns:
        dict with keys: reply, messageType, courses
    """
    text_reply = query_rag(message)
    return {
        "reply": text_reply,
        "messageType": "faq",
        "courses": None,
    }
