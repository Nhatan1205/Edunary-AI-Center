import argparse
from .get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from app.models.llm_config import llm

CHROMA_PATH = "chroma"

faq_prompt = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
model = llm


def query_rag(query_text: str):
    # load embeding function and db
    db = Chroma(
        collection_name="test_document",      
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function(),
    )
    # search the db: 
    results = db.similarity_search_with_score(query_text, k=5)
    
    # combine all chunks
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(faq_prompt)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # call LLM model
    response_result = model.invoke(prompt)
    # Handle both string (OllamaLLM) and AIMessage (ChatOpenAI, ChatGoogleGenerativeAI)
    response_text = response_result.content if hasattr(response_result, 'content') else response_result
    
    
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return response_text
