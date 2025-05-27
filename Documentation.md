# FinAI Project Documentation

## 📖 Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Services](#services)
- [API Documentation](#api-documentation)
- [Development Guide](#development-guide)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Project Overview

FinAI is an advanced multi-agent financial intelligence system designed to provide real-time market insights through natural language interaction. The system delivers spoken market briefings via an intuitive Streamlit interface, powered by a sophisticated backend of specialized AI agents.

## System Architecture

### Core Components

1. **API Agent**
   - **Port**: 8001
   - **Responsibilities**:
     - Fetching real-time and historical market data
     - Integration with financial data providers (AlphaVantage, Yahoo Finance)
     - Data normalization and preprocessing
   - **Tech Stack**: FastAPI, Pydantic, HTTPX

2. **Scraping Agent**
   - **Port**: 8002
   - **Responsibilities**:
     - Automated data collection from financial filings and news
     - Web scraping with rate limiting and error handling
     - Document processing and content extraction
   - **Tech Stack**: FastAPI, BeautifulSoup, LangChain, Unstructured

3. **Retriever Agent**
   - **Port**: 8003
   - **Responsibilities**:
     - Vector embeddings management
     - Semantic search implementation
     - Document chunking and embedding generation
   - **Tech Stack**: FAISS, Sentence Transformers, LangChain

4. **Streamlit UI**
   - **Port**: 8501
   - **Features**:
     - Interactive dashboard
     - Real-time data visualization
     - Voice interaction interface

## Services

### API Agent
- **Base URL**: `http://localhost:8001`
- **Endpoints**:
  - `GET /health` - Service health check
  - `GET /api/price/{symbol}` - Get current price
  - `GET /api/historical/{symbol}` - Get historical data

### Scraping Agent
- **Base URL**: `http://localhost:8002`
- **Endpoints**:
  - `GET /health` - Service health check
  - `GET /news` - Fetch financial news
  - `POST /filing` - Retrieve SEC filings

### Retriever Agent
- **Base URL**: `http://localhost:8003`
- **Endpoints**:
  - `GET /health` - Service health check
  - `POST /search` - Semantic search
  - `POST /ingest` - Ingest new documents

## Development Guide

### Prerequisites
- Python 3.11+
- Docker 20.10+
- Docker Compose 2.0+

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/epsilonstar-02/FinAI.git
   cd FinAI
   ```

2. Set up environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running Locally

1. Start all services with Docker:
   ```bash
   docker-compose up -d
   ```

2. Access services:
   - Streamlit UI: http://localhost:8501
   - API Documentation: http://localhost:8001/docs

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/agents/api_agent/test_main.py -v

# Run with coverage
pytest --cov=agents --cov-report=html
```

### Test Coverage

We aim to maintain at least 80% test coverage. Current coverage:
- API Agent: 85%
- Scraping Agent: 82%
- Retriever Agent: 78%

## Deployment

### Production Deployment

1. Update environment variables in `.env`
2. Build and start services:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d --build
   ```

### Monitoring

- **Logs**: `docker-compose logs -f`
- **Metrics**: Prometheus endpoint at `/metrics`
- **Health Checks**: `/health` endpoints on all services

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   - Ensure no other services are using ports 8001-8003 and 8501

2. **Missing Dependencies**
   - Run `docker-compose build --no-cache` to rebuild all images

3. **API Key Issues**
   - Verify all required API keys are set in `.env`

### Getting Help

For additional support:
1. Check the [GitHub Issues](https://github.com/epsilonstar-02/FinAI/issues)
2. Open a new issue with detailed error information

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
  - Mocking external API calls
  - Input validation testing
  - Error scenario coverage
- **Test Framework**
  - Pytest with pytest-asyncio for async tests
  - FastAPI TestClient for API testing
  - Fixtures for test data and mocks

## Technical Stack

### Backend Services
- **FastAPI**: For building high-performance microservices
- **Docker**: Containerization for consistent development and deployment
- **Docker Compose**: For orchestrating multi-container applications
- **Pydantic v2**: For data validation and settings management

### AI/ML Components
- **LangGraph/CrewAI**: For agent coordination and workflow management
- **Vector Databases**: FAISS/Pinecone for efficient similarity search
- **LLM Integration**: For natural language understanding and generation
- **Speech Processing**: Whisper for speech-to-text and TTS for text-to-speech

### Frontend
- **Streamlit**: For building interactive web interfaces
- **WebSockets**: For real-time communication between frontend and backend

## Project Structure
```
FinAI/
├── .dockerignore          # Docker ignore file
├── .env                   # Environment variables (not version controlled)
├── .env.example          # Example environment variables
├── Dockerfile.api         # API service Dockerfile
├── Dockerfile.streamlit   # Streamlit app Dockerfile
├── docker-compose.yml     # Docker Compose configuration
├── readme.md             # Project overview and setup instructions
├── requirements.txt       # Main project dependencies
├── requirements-api.txt   # API service dependencies
│
├── agents/              # Agent implementations
│   ├── api_agent/        # API Agent implementation
│   │   ├── __init__.py
│   │   ├── client.py     # API client implementation
│   │   ├── config.py     # Configuration settings
│   │   ├── main.py       # Main application entry point
│   │   ├── models.py     # Data models
│   │   └── requirements-api.txt  # Agent-specific dependencies
│   └── health/           # Health check endpoints
│       └── api_agent.py
│
└── streamlit_app/       # Streamlit application
    ├── app.py            # Main Streamlit application
    └── requirements-streamlit.txt  # Frontend dependencies
```

## Configuration

### Environment Variables
- `ALPHAVANTAGE_API_KEY`: API key for AlphaVantage service
- `GEMINI_API_KEY`: API key for Gemini (if used)

## Development Setup

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- API keys for required services

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and update with your API keys
4. Build and start services: `docker-compose up -d`

## Deployment

The application is designed to be deployed using Docker containers. The `docker-compose.yml` file defines the following services:

1. **api**: The FastAPI backend service
2. **streamlit**: The Streamlit frontend interface

## API Documentation

### Endpoints

#### Health Check
- `GET /health`: Returns the health status of the API service

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Future Enhancements

- Implement additional data providers
- Add more sophisticated analysis capabilities
- Enhance voice interaction features
- Implement user authentication and authorization
- Add support for custom alerts and notifications