# Language Agent

## Overview
The Language Agent is a microservice in the FinAI multi-agent system that generates natural language text using Google's Gemini Flash model. It provides a FastAPI interface for generating text based on financial data and context.

## Features
- Text generation using Google's Gemini Flash LLM
- Template-based prompting with Jinja2
- Market brief generation with financial context
- Async API with proper error handling
- Configurable model parameters

## Architecture
The Language Agent follows a modular design with the following components:

- **config.py**: Configuration settings using Pydantic
- **models.py**: Request/response data models
- **llm_client.py**: Google Generative AI client integration
- **prompts/**: Jinja2 templates for text generation
- **main.py**: FastAPI application with endpoints

## API Endpoints

### Health Check
- **Endpoint**: `GET /health`
- **Response**: `{"status": "ok", "agent": "Language Agent"}`

### Generate Text
- **Endpoint**: `POST /generate`
- **Request Body**:
  ```json
  {
    "query": "What's the market outlook for AAPL?",
    "context": {
      "prices": "AAPL: $190.25 (+1.2%), MSFT: $420.10 (-0.5%)",
      "news": "Apple announces new product line.",
      "chunks": "Apple's revenue increased by 10% YoY.",
      "analysis": "PE Ratio: 30.5, EPS: 6.24"
    }
  }
  ```
- **Response**:
  ```json
  {
    "text": "Apple's stock is showing positive momentum with a 1.2% gain, outperforming Microsoft which is down 0.5%. The recent product line announcement appears to be driving investor confidence, supported by strong fundamentals including a 10% year-over-year revenue growth. With a PE ratio of 30.5 and EPS of 6.24, Apple continues to demonstrate solid financial health despite broader market uncertainties."
  }
  ```

## Configuration
The Language Agent requires the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| GEMINI_API_KEY | Google Cloud API key for Gemini access | (Required) |
| GEMINI_MODEL | Gemini model name | "gemini-flash" |
| TIMEOUT | Request timeout in seconds | 10 |

## Deployment
The Language Agent is containerized using Docker and integrated with the FinAI system through docker-compose. It runs on port 8004 internally and is exposed on port 8005.

## Integration
The Language Agent is integrated with the Orchestrator service, which coordinates the workflow between multiple agents. The Orchestrator communicates with the Language Agent through the URL specified in the `LANGUAGE_AGENT_URL` environment variable.

## Development
To run the Language Agent locally for development:

```bash
# Install dependencies
pip install -r agents/language_agent/requirements-language.txt

# Set environment variables
export GEMINI_API_KEY=your_api_key
export GEMINI_MODEL=gemini-flash
export TIMEOUT=10

# Run the service
uvicorn agents.language_agent.main:app --reload
```

## Testing
Unit tests are available in the `tests/agents/language_agent/` directory and can be run with pytest:

```bash
pytest tests/agents/language_agent/
```
