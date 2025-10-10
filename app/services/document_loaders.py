from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
import os

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "document/edunary")


def generate_data_store():
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    return loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)



def add_to_chroma(chunks: list[Document]):
    # Load (or create new) database
    db = Chroma(
        collection_name="test_document",        # name the collection
        persist_directory=CHROMA_PATH,          # archive folder
        embedding_function=get_embedding_function(), #embedding function to convert text → vector.
    )

    # Generate ID for each chunk
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Get existing IDs in DB
    existing_items = db.get(include=[])  # always have 'ids' field
    existing_ids = set(existing_items["ids"])

    # Add only new documents
    new_chunks = []
    new_ids = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk.metadata["id"])

    if new_chunks:
        db.add_documents(new_chunks, ids=new_ids)
        
def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

if __name__ == "__main__":
    generate_data_store()
    
# chạy "python .\app\services\document_loaders.py"