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
    """
    if not os.path.exists(VECTOR_PATH):
        # Return a dummy or handle empty state if directory doesn't exist yet
        print("[WARNING] Vector path does not exist. Please upload documents first.")
        return None

    # Load Chroma vectorstore
    vectordb = Chroma(
        persist_directory=VECTOR_PATH, 
        embedding_function=EMBEDDINGS
    )

    # Fetch all documents currently in the DB to build the BM25 index
    # This ensures BM25 search only looks at the currently 'ingested' files
    db_data = vectordb.get()
    
    all_docs = []
    if db_data["documents"]:
        for i in range(len(db_data["documents"])):
            all_docs.append(Document(
                page_content=db_data["documents"][i],
                metadata=db_data["metadatas"][i] if db_data["metadatas"] else {}
            ))

    # Initialize the Hybrid logic
    hybrid = HybridRetriever(vectordb, all_docs, k=5)

    return hybrid

# # retriever.py
# from langchain_chroma import Chroma
# from config import EMBEDDINGS, VECTOR_PATH
# from langchain_core.documents import Document
# from rank_bm25 import BM25Okapi


# class BM25RetrieverCustom:
#     """
#     Simple BM25 retriever that works on a list of Document objects.
#     """
#     def __init__(self, docs, k=4):
#         # Ensure all docs are Document objects
#         self.docs = [d if isinstance(d, Document) else Document(page_content=str(d), metadata={}) for d in docs]
#         # tokenize corpus for BM25
#         self.tokenized_corpus = [d.page_content.split() for d in self.docs]
#         self.bm25 = BM25Okapi(self.tokenized_corpus)
#         self.k = k

#     def get_relevant_documents(self, query):
#         query_tokens = query.split()
#         scores = self.bm25.get_scores(query_tokens)
#         top_n_idx = scores.argsort()[-self.k:][::-1]
#         results = [self.docs[i] for i in top_n_idx]
#         print(f"[BM25] Top {self.k} docs selected (by index): {top_n_idx}")
#         return results


# class HybridRetriever:
#     """
#     Hybrid Ensemble Retriever combining BM25 + Vector retriever
#     """
#     def __init__(self, vectorstore, bm25_docs, k=4, bm25_weight=0.5, vector_weight=0.5):
#         self.vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
#         self.bm25_retriever = BM25RetrieverCustom(bm25_docs, k=k)
#         self.k = k
#         self.bm25_weight = bm25_weight
#         self.vector_weight = vector_weight

#     def get_relevant_documents(self, query):
#         # BM25 results
#         bm25_docs = self.bm25_retriever.invoke(query)
#         # Vector results
#         vector_docs = self.vector_retriever.invoke(query)

#         # For debugging, print which doc comes from which retriever
#         print("[Hybrid] BM25 docs:")
#         for d in bm25_docs:
#             print(f" - {d.page_content[:50]}...")

#         print("[Hybrid] Vector docs:")
#         for d in vector_docs:
#             print(f" - {d.page_content[:50]}...")

#         # Simple merge by weight (for now, just interleave)
#         combined = bm25_docs[:int(self.k*self.bm25_weight)] + vector_docs[:int(self.k*self.vector_weight)]
#         return combined[:self.k]  # final top-k


# def get_retriever():
#     # Load Chroma vectorstore
#     vectordb = Chroma(
#     persist_directory=VECTOR_PATH, 
#     embedding_function=EMBEDDINGS
# )

#     # Get all documents from vectorstore for BM25
#     docs = vectordb.get()["documents"]
#     # Ensure Document objects
#     all_docs = [d if isinstance(d, Document) else Document(page_content=str(d), metadata={}) for d in docs]

#     # Create hybrid retriever
#     hybrid = HybridRetriever(vectordb, all_docs, k=4, bm25_weight=0.5, vector_weight=0.5)

#     return hybrid