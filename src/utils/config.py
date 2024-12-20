"""
This module manages configuration settings for the application by
loading environment variables from `.env` files. It defines key settings
for OpenAI and Langfuse integrations, as well as paths for default directories
 used throughout the application.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Together AI
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Langfuse
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Docs
DOCS_FOLDER = "src/docs"

DEFAULT_USER_ID = "default_user"
DEFAULT_THREAD_ID = "default_thread"
PRIMARY_LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
BACKUP_LLM_MODEL = "gpt-4o-mini"
