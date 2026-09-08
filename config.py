import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================
# LLM
# ============================================================

LLM = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0,
    max_tokens=500,
    reasoning_effort="none",
    reasoning_format="hidden",
    groq_api_key=GROQ_API_KEY
)

# ============================================================
# EMBEDDINGS
# ============================================================

EMBEDDINGS = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

# ============================================================
# CHUNKING
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ============================================================
# VECTOR DATABASE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_PATH = os.path.join(
    BASE_DIR,
    "vectorstore",
    "chroma_db"
)