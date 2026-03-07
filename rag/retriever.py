# rag/retriever.py
from langchain_chroma import Chroma
from config import EMBEDDINGS, VECTOR_PATH
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import os

class HybridRetriever:
    """
    Hybrid Retriever combining Keyword (BM25) and Semantic (Vector) search.
    Uses .invoke() to comply with LangChain 2026 standards.
    """
    def __init__(self, vectorstore, docs, k=4):
        # 1. Setup Vector Retriever
        self.vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        
        # 2. Setup BM25 Retriever
        # Ensure we have a list of Document objects
        self.docs = [Document(page_content=d) if isinstance(d, str) else d for d in docs]
        
        if self.docs:
            # Tokenize corpus for BM25
            tokenized_corpus = [d.page_content.split() for d in self.docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None
            
        self.k = k

    def invoke(self, query: str):
        """
        Main retrieval method. Replaces deprecated get_relevant_documents.
        """
        # --- 1. Vector Search (Semantic) ---
        vector_docs = self.vector_retriever.invoke(query)
        
        # --- 2. BM25 Search (Keyword) ---
        bm25_docs = []
        if self.bm25:
            query_tokens = query.split()
            # Get scores for all docs and take top N
            doc_scores = self.bm25.get_scores(query_tokens)
            top_n_indices = doc_scores.argsort()[-self.k:][::-1]
            bm25_docs = [self.docs[i] for i in top_n_indices if doc_scores[i] > 0]

        # --- 3. Hybrid RRF (Reciprocal Rank Fusion) / Merging ---
        # Combine lists and remove duplicates based on content
        seen_content = set()
        combined_docs = []
        
        # Interleave results (Vector first, then BM25)
        for doc in (vector_docs + bm25_docs):
            if doc.page_content not in seen_content:
                combined_docs.append(doc)
                seen_content.add(doc.page_content)
        
        print(f"[RETRIEVER] Found {len(vector_docs)} vector and {len(bm25_docs)} keyword matches.")
        return combined_docs[:self.k]

def get_retriever():
    """
    Factory function to initialize and return the Hybrid Retriever.
    Safely handles empty or missing vector stores on Render.
    """
    # 1. Path Safety Check
    if not os.path.exists(VECTOR_PATH) or not os.listdir(VECTOR_PATH):
        print(f"[SYSTEM] Vector store path {VECTOR_PATH} is missing or empty.")
        return None

    try:
        # 2. Load Chroma vectorstore
        vectordb = Chroma(
            persist_directory=VECTOR_PATH, 
            embedding_function=EMBEDDINGS
        )

        # 3. Fetch data to rebuild BM25 index
        db_data = vectordb.get()
        
        # 4. Content Safety Check: folder exists but might be a corrupted/empty DB
        if not db_data or not db_data.get("documents"):
            print("[SYSTEM] Vector store exists but contains no documents.")
            return None
        
        all_docs = []
        for i in range(len(db_data["documents"])):
            all_docs.append(Document(
                page_content=db_data["documents"][i],
                metadata=db_data["metadatas"][i] if db_data["metadatas"] else {}
            ))

        # 5. Initialize Hybrid Retriever
        # (Ensure k=5 matches your memory constraints on Render)
        hybrid = HybridRetriever(vectordb, all_docs, k=5)
        return hybrid

    except Exception as e:
        print(f"[SYSTEM] Critical Error initializing retriever: {e}")
        return None