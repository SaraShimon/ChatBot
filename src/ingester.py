from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

from src.config import PDF_DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS_MODEL

def index_documents(vector_store_instance: InMemoryVectorStore) -> None:
    """
    Loads PDF documents from a specified directory, splits them into chunks,
    and indexes them into the provided vector store.

    Args:
        vector_store_instance: An initialized InMemoryVectorStore instance.
    """
    print(f"Loading documents from {PDF_DATA_PATH}...")
    loader = PyPDFDirectoryLoader(PDF_DATA_PATH)
    docs = loader.load()

    print(f"Splitting {len(docs)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(docs)

    print(f"Adding {len(all_splits)} chunks to the vector store...")
    vector_store_instance.add_documents(documents=all_splits)
    print("Document indexing complete.")

# Global instance of vector store (initialized once)
# For InMemoryVectorStore, it needs to be initialized and indexed only once.
vector_store = InMemoryVectorStore(EMBEDDINGS_MODEL)
index_documents(vector_store)
