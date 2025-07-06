"""
LLM Configuration Module - Centralized configuration for language models
Supports multiple providers: Groq (default), OpenAI, and Ollama
"""

import os
import logging
from typing import Dict, List, Any, Optional, Literal, Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    provider: Literal["groq", "openai", "ollama"] = Field(default="groq")
    model_name: str = Field(default="gemma2-9b-it")
    temperature: float = Field(default=0.2)
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    context_window: Optional[int] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)

    @validator('api_key', pre=True, always=True)
    def validate_api_key(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate and get API key from environment if not provided"""
        if v is not None:
            return v
            
        provider = values.get('provider', 'groq')
        env_key = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(env_key)
        
        if not api_key and provider != 'ollama':  # Ollama doesn't require API key
            logger.warning(f"No API key found for {provider} in environment variables")
        return api_key

    @validator('base_url', pre=True, always=True)
    def validate_base_url(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate and get base URL from environment if not provided"""
        if v is not None:
            return v
            
        provider = values.get('provider', 'groq')
        env_url = f"{provider.upper()}_BASE_URL"
        base_url = os.getenv(env_url)
        
        if not base_url:
            if provider == 'ollama':
                base_url = "http://localhost:11434"  # Default Ollama URL
            elif provider == 'openai':
                base_url = "https://api.openai.com/v1"  # Default OpenAI URL
            elif provider == 'groq':
                base_url = "https://api.groq.com"  # Default Groq URL
                
        return base_url

def get_default_config() -> LLMConfig:
    """Get default LLM configuration based on environment variables"""
    
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider not in ["groq", "openai", "ollama"]:
        logger.warning(f"Invalid LLM provider {provider}, defaulting to groq")
        provider = "groq"
        
    return LLMConfig(
        provider=provider,
        model_name=os.getenv("LLM_MODEL_NAME", "gemma2-9b-it"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )

def initialize_llm(config: LLMConfig) -> BaseChatModel:
    """Initialize LLM based on provider configuration"""
    
    try:
        if config.provider == "openai":
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url
            )
        elif config.provider == "groq":
            return ChatGroq(
                model_name=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url
            )
        elif config.provider == "ollama":
            return ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
                base_url=config.base_url
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
            
    except Exception as e:
        logger.error(f"Error initializing {config.provider} LLM: {str(e)}")
        raise

def validate_configurations(config: LLMConfig, fallback_providers: Optional[List[LLMConfig]] = None) -> None:
    """Validate all LLM configurations"""
    # Validate primary configuration
    if not config.api_key and config.provider != 'ollama':
        raise ValueError(
            f"No API key found for {config.provider}. "
            f"Please set {config.provider.upper()}_API_KEY in .env file"
        )
    
    # Validate fallback configurations
    if fallback_providers:
        for fallback_config in fallback_providers:
            if not fallback_config.api_key and fallback_config.provider != 'ollama':
                raise ValueError(
                    f"No API key found for fallback provider {fallback_config.provider}. "
                    f"Please set {fallback_config.provider.upper()}_API_KEY in .env file"
                )

# Example usage
if __name__ == "__main__":
    # Test default configuration
    default_config = get_default_config()
    print("Default Config:", default_config.dict())
    
    # Test custom configuration
    custom_config = LLMConfig(
        provider="openai",
        model_name="gpt-4-turbo-preview",
        temperature=0.3
    )
    print("\nCustom Config:", custom_config.dict())
    
    # Test fallback configuration
    fallback_config = LLMConfig(
        provider="ollama",
        model_name="mistral",
        temperature=0.2
    )
    print("\nFallback Config:", fallback_config.dict()) 