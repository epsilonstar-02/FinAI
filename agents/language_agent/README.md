# Language Agent

## Overview
The Language Agent is a microservice designed for the FinAI multi-agent system. It provides a flexible and robust natural language text generation capability by supporting multiple Large Language Model (LLM) providers. It features fallback mechanisms, caching, and template-based prompting to generate financial insights, market briefs, or other text formats based on provided context.

## Features
- **Multi-Provider LLM Support:** Integrates with various LLM providers:
    - Google Gemini
    - OpenAI (GPT models)
    - Anthropic (Claude models)
    - Local Llama models (via LlamaCpp)
    - HuggingFace models (local inference via `transformers` or via Inference API)
- **Configurable Provider Priority:** Define a default provider and a list of fallback providers.
- **Template-Based Prompting:** Uses Jinja2 for dynamic prompt construction.
- **Response Caching:** Caches LLM responses to reduce latency and cost, with configurable TTL.
- **Retry Mechanisms:** Implements retries with exponential backoff for individual provider calls.
- **Per-Request Parameter Overrides:** Allows overriding default generation parameters (e.g., `max_tokens`, `temperature`) for specific requests.
- **Asynchronous API:** Built with FastAPI for high performance.
- **Comprehensive Configuration:** Settings managed via Pydantic and environment variables.

## Architecture
The Language Agent is structured as follows:
- **`config.py`**: Pydantic-based configuration settings, loaded from environment variables and `.env` files.
- **`models.py`**: Pydantic models for API request and response validation.
- **`multi_llm_client.py`**: The core client for interacting with various LLM providers. Manages provider selection, retries, caching, and API calls.
- **`prompts/`**: Directory containing Jinja2 templates for prompt engineering (e.g., `market_brief.tpl`).
- **`main.py`**: FastAPI application defining API endpoints, middleware, and request handling.

## API Endpoints

### Health Check
- **Endpoint**: `GET /health`
- **Description**: Checks the operational status of the agent, lists available providers, default provider, and loaded templates.
- **Response Example**:
  ```json
  {
    "status": "ok",
    "agent": "Language Agent",
    "version": "0.3.0",
    "timestamp": "2023-10-27T10:00:00.000Z",
    "providers": ["llama", "huggingface", "openai"],
    "default_provider": "llama",
    "templates": ["market_brief", "custom_summary"]
  }