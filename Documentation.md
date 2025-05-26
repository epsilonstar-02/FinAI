# FinAI Project Documentation

## Project Overview
FinAI is an advanced multi-agent financial intelligence system designed to provide real-time market insights through natural language interaction. The system delivers spoken market briefings via an intuitive Streamlit interface, powered by a sophisticated backend of specialized AI agents.

## System Architecture

### Core Components

1. **API Agent**
   - Responsible for fetching real-time and historical market data
   - Implements integration with financial data providers (AlphaVantage, Yahoo Finance)
   - Handles data normalization and preprocessing

2. **Scraping Agent**
   - Automates data collection from financial filings and news sources
   - Implements efficient web scraping with Python loaders
   - Handles rate limiting and error handling

3. **Retriever Agent**
   - Manages vector embeddings using FAISS or Pinecone
   - Implements semantic search for relevant financial information
   - Handles document chunking and embedding generation

4. **Analysis Agent**
   - Processes and interprets financial data
   - Identifies trends, anomalies, and key metrics
   - Generates insights and summaries from raw data

5. **Language Agent**
   - Generates human-like financial narratives
   - Utilizes advanced LLM capabilities for coherent reporting
   - Handles context management and response generation

6. **Voice Agent**
   - Manages speech-to-text (Whisper) and text-to-speech conversion
   - Handles voice I/O pipelines
   - Implements voice activity detection and audio processing

## Technical Stack

### Backend Services
- **FastAPI**: For building high-performance microservices
- **Docker**: Containerization for consistent development and deployment
- **Docker Compose**: For orchestrating multi-container applications

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