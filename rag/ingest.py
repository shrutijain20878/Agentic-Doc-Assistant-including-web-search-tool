import os
import gc  # Garbage Collector
from langchain_chroma import Chroma
from config import EMBEDDINGS, VECTOR_PATH
from utils.file_loader import load_pdf
from utils.chunking import chunk_documents

def ingest_files(files):
    """
    Memory-efficient ingestion that clears data after every file
    to prevent Render memory crashes.
    """
    # 1. Initialize the vectorstore first
    vectorstore = Chroma(
        persist_directory=VECTOR_PATH,
        embedding_function=EMBEDDINGS
    )

    # 2. CLEAR OLD DATA (Optimized: only fetch IDs, not documents)
    existing_data = vectorstore.get(include=[]) 
    if existing_data and existing_data.get("ids"):
        print(f"[INGEST] Clearing {len(existing_data['ids'])} old records...")
        vectorstore.delete(ids=existing_data["ids"])

    # 3. ADD NEW DATA FILE-BY-FILE
    for file in files:
        file_name = getattr(file, 'name', 'uploaded_file')
        print(f"[INGEST] Processing: {file_name}")

        # Load & Chunk
        docs = load_pdf(file)
        for doc in docs:
            doc.metadata["source_file"] = file_name
        
        chunks = chunk_documents(docs)
        
        # Add to DB immediately
        if chunks:
            vectorstore.add_documents(chunks)
        
        # --- THE CRITICAL FIX: Memory Management ---
        del docs
        del chunks
        gc.collect() # Force Python to free up RAM now
        # --------------------------------------------

    print("[INGEST] All files processed and persisted.")
    return vectorstore