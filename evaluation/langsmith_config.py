import os
from langsmith import Client
from dotenv import load_dotenv

def setup_langsmith():
    """Initialize LangSmith client and environment."""
    load_dotenv()
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        raise ValueError("LANGSMITH_API_KEY not set in environment variables.")
    client = Client(api_key=api_key)
    return client 