# config.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings

# This looks for the .env file and loads the variables
load_dotenv()

# 1. LLM Configuration (Switching from Ollama to Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )
# else:
#     # Fallback for local testing if you still want to use Ollama
#     from langchain_ollama import ChatOllama
#     LLM = ChatOllama(model="llama3", temperature=0)

# 2. Embeddings Configuration
# Switching to HuggingFace so it works in the cloud without Ollama
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HF_TOKEN"), 
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# 3. Chunking & Storage
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# Important: On Render, use a path the app can write to
VECTOR_PATH = "vectorstore/chroma_db"

# from langchain_ollama import ChatOllama
# from langchain_ollama import OllamaEmbeddings

# # LLM
# LLM = ChatOllama(
#     model="llama3",
#     temperature=0
# )

# # Embeddings
# EMBEDDINGS = OllamaEmbeddings(
#     model="nomic-embed-text"
# )

# # Chunking configuration
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200
# VECTOR_PATH = "vectorstore/chroma_db"