import os
from langchain_chroma import Chroma
from config import EMBEDDINGS, VECTOR_PATH
from utils.file_loader import load_pdf
from utils.chunking import chunk_documents

def ingest_files(files):
    all_chunks = []

    for file in files:
        docs = load_pdf(file)
        # Add metadata so we can delete specifically these files later if needed
        for doc in docs:
            # Use getattr or check if 'file' has a name attribute to prevent crashes
            doc.metadata["source_file"] = getattr(file, 'name', 'uploaded_file')
        
        chunks = chunk_documents(docs)
        all_chunks.extend(chunks)

    # 1. Initialize the vectorstore
    vectorstore = Chroma(
        persist_directory=VECTOR_PATH,
        embedding_function=EMBEDDINGS
    )

    # 2. CLEAR OLD DATA (The safe way)
    # We use .get() to retrieve all internal IDs currently in the DB
    existing_data = vectorstore.get()
    if existing_data and "ids" in existing_data and existing_data["ids"]:
        # Only attempt delete if there are actual IDs present
        vectorstore.delete(ids=existing_data["ids"])

    # 3. ADD NEW DATA
    if all_chunks:
        vectorstore.add_documents(all_chunks)
    
    # In LangChain 2026, persistence is automatic, 
    # but returning the object is good practice.
    return vectorstore