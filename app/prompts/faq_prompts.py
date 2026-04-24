"""
FAQ / RAG prompt templates.
Extracted from query_data.py.
"""

# RAG prompt — answers questions using only the retrieved context.
FAQ_RAG_PROMPT = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
