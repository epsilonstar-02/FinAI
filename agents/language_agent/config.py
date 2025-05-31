# agents/language_agent/config.py
# Primarily adding LOG_LEVEL application and minor path check improvements.

from pydantic_settings import BaseSettings, SettingsConfigDict # Added SettingsConfigDict for Pydantic V2
from dotenv import load_dotenv
from enum import Enum
from typing import Optional, Dict, List, Any, Union
import os
from pathlib import Path
import logging # Added for log level validation

load_dotenv()

class LLMProvider(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LLAMA = "llama"
    HUGGINGFACE = "huggingface"
    # LANGCHAIN provider was in original enum but not used in client. Keeping for now.
    LANGCHAIN = "langchain"


class Settings(BaseSettings):
    DEFAULT_PROVIDER: LLMProvider = LLMProvider.LLAMA
    FALLBACK_PROVIDERS: List[LLMProvider] = [LLMProvider.HUGGINGFACE]
    TIMEOUT: int = 30 # Default timeout for provider API calls
    MAX_RETRIES: int = 2 # Reduced from 3 for faster fallback in some cases
    RETRY_DELAY: int = 2 # Base delay seconds for exponential backoff
    CACHE_RESPONSES: bool = True
    CACHE_TTL: int = 3600
    LOG_LEVEL: str = "INFO"
    
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.2
    TOP_P: float = 0.95
    TOP_K: int = 40 # Used by some providers like Gemini, HF
    
    TEMPLATES_DIR: str = "prompts" # Relative to the package/module containing this config
    
    # Llama specific
    LLAMA_MODEL_PATH: str = os.getenv("LLAMA_MODEL_PATH", "./models/llama/ggml-model-q4_0.bin") # Allow override
    LLAMA_N_CTX: int = 2048
    LLAMA_N_BATCH: int = 512
    LLAMA_USE_GPU: bool = False # Default to CPU for wider compatibility
    
    # HuggingFace specific
    HUGGINGFACE_API_KEY: Optional[str] = None
    HUGGINGFACE_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"
    HUGGINGFACE_QUANTIZE: Optional[str] = None # was "4bit", None is safer default if bitsandbytes not set up
    HUGGINGFACE_USE_LOCAL: bool = True
    
    # Paid/Commercial - API keys default to None
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_ORGANIZATION: Optional[str] = None
    
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-sonnet-20240229" # Example, claude-instant is cheaper
    
    # LangChain settings (if used more broadly)
    LANGCHAIN_CACHE: bool = False # Defaulting to False as it might have its own cache system.

    model_config = SettingsConfigDict( # Pydantic V2 style
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    def _resolve_model_path(self, path_str: Optional[str]) -> Optional[Path]:
        """Helper to resolve model paths relative to a base if needed, or ensure they exist."""
        if not path_str:
            return None
        # Assuming paths in .env are absolute or relative to project root.
        # For now, just convert to Path object.
        # More sophisticated logic could try to resolve relative to a project base dir.
        p = Path(path_str)
        if p.exists():
            return p
        # Try resolving relative to a common "models" directory at project root if path is simple name
        # This is just an example, real path resolution might need more context.
        # For now, we'll rely on the path being correctly specified in env or default.
        # logger.warning(f"Model path {path_str} does not exist as specified.")
        return p # Return Path object even if it doesn't exist, check is done in is_provider_available


    def is_provider_available(self, provider: LLMProvider) -> bool:
        if provider == LLMProvider.GOOGLE:
            return bool(self.GEMINI_API_KEY)
        elif provider == LLMProvider.OPENAI:
            return bool(self.OPENAI_API_KEY)
        elif provider == LLMProvider.ANTHROPIC:
            return bool(self.ANTHROPIC_API_KEY)
        elif provider == LLMProvider.LLAMA:
            llama_path = self._resolve_model_path(self.LLAMA_MODEL_PATH)
            return bool(llama_path and llama_path.is_file())
        elif provider == LLMProvider.HUGGINGFACE:
            if self.HUGGINGFACE_USE_LOCAL:
                # Availability depends on transformers/pytorch and valid model name.
                # Actual loading error caught at runtime. Assume available if flag is set and model name provided.
                return bool(self.HUGGINGFACE_MODEL)
            return bool(self.HUGGINGFACE_API_KEY) # For API access
        elif provider == LLMProvider.LANGCHAIN:
            # LangChain itself is a framework.
            # Consider it "available" if it's installed and at least one underlying model it could use is configured.
            # This check is simplified; real LangChain setup is more complex.
            return any(self.is_provider_available(p) for p in LLMProvider if p != LLMProvider.LANGCHAIN)
        return False
    
    def get_available_providers(self) -> List[LLMProvider]:
        return [p for p in LLMProvider if self.is_provider_available(p)]

settings = Settings()

# Configure logging
log_level_to_set = settings.LOG_LEVEL.upper()
if not hasattr(logging, log_level_to_set):
    # Use a temporary logger for this warning if main app logging isn't set up yet
    temp_logger = logging.getLogger(__name__)
    temp_logger.warning(f"Invalid LOG_LEVEL '{log_level_to_set}' in Language Agent settings. Defaulting to INFO.")
    log_level_to_set = "INFO"

logging.basicConfig(
    level=getattr(logging, log_level_to_set),
    format="%(asctime)s - %(name)s (LANG_AGENT) - %(levelname)s - %(message)s"
)
# Re-get logger after basicConfig
logger = logging.getLogger(__name__)

# Check Llama model path after logger is configured
if settings.DEFAULT_PROVIDER == LLMProvider.LLAMA or LLMProvider.LLAMA in settings.FALLBACK_PROVIDERS:
    if not settings.is_provider_available(LLMProvider.LLAMA):
        logger.warning(
            f"Llama provider is configured as default or fallback, but model path "
            f"'{settings.LLAMA_MODEL_PATH}' does not exist or is not a file. Llama will be unavailable."
        )