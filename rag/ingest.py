import os
import gc

from langchain_chroma import Chroma

from config import EMBEDDINGS, VECTOR_PATH
from utils.file_loader import load_pdf
from utils.chunking import chunk_documents


def ingest_files(files):
    """
    Ingest uploaded PDF files into ChromaDB.

    Existing documents are cleared before adding the newly
    uploaded documents.
    """

    # ---------------------------------------------------------
    # Ensure ChromaDB directory exists
    # ---------------------------------------------------------

    os.makedirs(
        VECTOR_PATH,
        exist_ok=True
    )

    print(
        f"[INGEST] ChromaDB path: {VECTOR_PATH}"
    )

    # ---------------------------------------------------------
    # Initialize ChromaDB
    # ---------------------------------------------------------

    vectorstore = Chroma(
        persist_directory=VECTOR_PATH,
        embedding_function=EMBEDDINGS
    )

    # ---------------------------------------------------------
    # Clear existing documents
    # ---------------------------------------------------------

    try:

        existing_data = vectorstore.get(
            include=[]
        )

        existing_ids = existing_data.get(
            "ids",
            []
        )

        if existing_ids:

            print(
                f"[INGEST] Clearing "
                f"{len(existing_ids)} old records..."
            )

            vectorstore.delete(
                ids=existing_ids
            )

    except Exception as e:

        print(
            f"[WARNING] Could not clear existing "
            f"ChromaDB records: {e}"
        )

    # ---------------------------------------------------------
    # Process PDFs one by one
    # ---------------------------------------------------------

    for uploaded_file in files:

        file_name = getattr(
            uploaded_file,
            "name",
            "uploaded_file.pdf"
        )

        print(
            f"[INGEST] Processing: {file_name}"
        )

        docs = []
        chunks = []

        try:

            # ---------------------------------------------
            # Load PDF
            # ---------------------------------------------

            docs = load_pdf(
                uploaded_file
            )

            # ---------------------------------------------
            # Add source metadata
            # ---------------------------------------------

            for doc in docs:

                doc.metadata[
                    "source_file"
                ] = file_name

            # ---------------------------------------------
            # Split into chunks
            # ---------------------------------------------

            chunks = chunk_documents(
                docs
            )

            # ---------------------------------------------
            # Add chunks to ChromaDB
            # ---------------------------------------------

            if chunks:

                vectorstore.add_documents(
                    chunks
                )

                print(
                    f"[INGEST] Added "
                    f"{len(chunks)} chunks "
                    f"from {file_name}"
                )

        except Exception as e:

            print(
                f"[ERROR] Failed to process "
                f"{file_name}: {e}"
            )

            raise

        finally:

            # ---------------------------------------------
            # Memory cleanup
            # ---------------------------------------------

            del docs
            del chunks

            gc.collect()

    print(
        "[INGEST] All files processed successfully."
    )

    return vectorstore