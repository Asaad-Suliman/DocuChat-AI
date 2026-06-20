import os
from dotenv import load_dotenv

load_dotenv(override=True)

# OpenAI API key
OPENAI_API_KEY = os.getenv("GPT_KEY", "your-key-here")

# Model settings
MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Document chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.7

# Supported file types
SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt"]