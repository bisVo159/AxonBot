import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME=os.getenv("PINECONE_INDEX_NAME","axonbot-index")
DOC_SOURCE_DIR=os.getenv("DOC_SOURCE_DIR","data")
EMBED_MODEL=os.getenv("EMBED_MODEL","sentence-transformers/all-mpnet-base-v2")