from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from enum import Enum
from typing import Optional, Dict, List, Any, Union
import os
from pathlib import Path

# Load environment variables from .env file
load_dotenv()


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GOOGLE = "google"        # Google Generative AI (Gemini)
    OPENAI = "openai"        # OpenAI (GPT models)
    ANTHROPIC = "anthropic"  # Anthropic (Claude models)
    LLAMA = "llama"          # Local Llama models
    HUGGINGFACE = "huggingface"  # HuggingFace models
    LANGCHAIN = "langchain"  # LangChain for model orchestration


class Settings(BaseSettings):
    """
    Configuration settings for the Language Agent with multi-provider support.
    
    Attributes:
        DEFAULT_PROVIDER: Default LLM provider to use
        FALLBACK_PROVIDERS: List of providers to try if default fails
        TIMEOUT: Request timeout in seconds
        CACHE_RESPONSES: Whether to cache LLM responses
        CACHE_TTL: Time-to-live for cached responses in seconds
        
        # Google settings
        GEMINI_API_KEY: API key for Google Generative AI
        GEMINI_MODEL: Model name for Gemini
        
        # OpenAI settings
        OPENAI_API_KEY: API key for OpenAI
        OPENAI_MODEL: OpenAI model to use
        
        # Anthropic settings
        ANTHROPIC_API_KEY: API key for Anthropic
        ANTHROPIC_MODEL: Anthropic model to use
        
        # Llama settings
        LLAMA_MODEL_PATH: Path to Llama model
        
        # HuggingFace settings
        HUGGINGFACE_API_KEY: API key for HuggingFace
        HUGGINGFACE_MODEL: HuggingFace model to use
        HUGGINGFACE_QUANTIZE: Quantization for local models
    """
    # Core settings - Prioritizing free and open-source options by default
    DEFAULT_PROVIDER: LLMProvider = LLMProvider.LLAMA         # Default to local, free Llama model
    FALLBACK_PROVIDERS: List[LLMProvider] = [LLMProvider.HUGGINGFACE]  # Fallback to HuggingFace (has free models)
    TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1
    CACHE_RESPONSES: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    LOG_LEVEL: str = "INFO"
    
    # Generation parameters
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.2
    TOP_P: float = 0.95
    TOP_K: int = 40
    
    # Templates path
    TEMPLATES_DIR: str = "prompts"
    
    # Free and Open-Source LLM settings
    # Llama settings - Local, free and open-source
    LLAMA_MODEL_PATH: str = "./models/llama/ggml-model-q4_0.bin"  # Default path to Llama model
    LLAMA_N_CTX: int = 2048
    LLAMA_N_BATCH: int = 512
    LLAMA_USE_GPU: bool = False
    
    # HuggingFace settings - Many free and open-source models available
    HUGGINGFACE_API_KEY: Optional[str] = None              # Optional, can use without API key
    HUGGINGFACE_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"  # Free and open-source model
    HUGGINGFACE_QUANTIZE: Optional[str] = "4bit"          # 4bit, 8bit, or None
    HUGGINGFACE_USE_LOCAL: bool = True                    # Use local models by default
    
    # NOTE: The following are paid/commercial API services and are OPTIONAL
    # If these API keys are not provided, system will use free and open-source alternatives only
    
    # Google Generative AI settings - Paid service
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-flash"
    
    # OpenAI settings - Paid service
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_ORGANIZATION: Optional[str] = None
    
    # Anthropic settings - Paid service
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-sonnet-20240229"
    
    # LangChain settings
    LANGCHAIN_CACHE: bool = True
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
    
    def is_provider_available(self, provider: LLMProvider) -> bool:
        """Check if a provider is available based on API keys"""
        if provider == LLMProvider.GOOGLE:
            return bool(self.GEMINI_API_KEY)
        elif provider == LLMProvider.OPENAI:
            return bool(self.OPENAI_API_KEY)
        elif provider == LLMProvider.ANTHROPIC:
            return bool(self.ANTHROPIC_API_KEY)
        elif provider == LLMProvider.LLAMA:
            return bool(self.LLAMA_MODEL_PATH) and Path(self.LLAMA_MODEL_PATH).exists()
        elif provider == LLMProvider.HUGGINGFACE:
            if self.HUGGINGFACE_USE_LOCAL:
                return True
            return bool(self.HUGGINGFACE_API_KEY)
        elif provider == LLMProvider.LANGCHAIN:
            # LangChain requires at least one other provider
            return any(self.is_provider_available(p) for p in [
                LLMProvider.GOOGLE, LLMProvider.OPENAI, 
                LLMProvider.ANTHROPIC, LLMProvider.LLAMA,
                LLMProvider.HUGGINGFACE
            ])
        return False
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers"""
        return [p for p in LLMProvider if self.is_provider_available(p)]


# Create settings instance
settings = Settings()
