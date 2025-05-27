# FinAI: Intelligent Financial Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-2CA5E0?logo=docker&logoColor=white)](https://www.docker.com/)

## Overview
FinAI is an advanced, multi-agent financial intelligence system designed to provide real-time market insights through natural language interaction. The system delivers spoken market briefings via an intuitive Streamlit interface, powered by a sophisticated backend of specialized AI agents.

## 🚀 Key Features

- **Real-time Market Analysis**: Get up-to-the-minute financial data and insights
- **Voice-Activated Interface**: Natural language interaction through speech-to-text and text-to-speech
- **Multi-Agent Architecture**: Specialized agents working in harmony to process and analyze financial data
- **Retrieval-Augmented Generation (RAG)**: Combines real-time data with contextual knowledge
- **Containerized Deployment**: Easy setup with Docker and Docker Compose
- **Scalable & Secure**: Built with security and scalability in mind

## 🏗️ System Architecture

```
FinAI
├── agents/
│   ├── api_agent/         # Handles financial data API integrations
│   ├── scraping_agent/    # Web scraping for financial data
│   ├── retriever_agent/   # Vector embeddings and semantic search
│   └── language_agent/    # Text generation with Google Gemini
├── streamlit_app/         # Web interface
├── data_ingestion/        # Data processing pipelines
└── orchestrator/          # Coordinates agent interactions
```

## 🛠️ Installation

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (for local development)

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/epsilonstar-02/FinAI.git
   cd FinAI
   ```

2. Copy the example environment file and update with your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Access the application:
   - Streamlit UI: http://localhost:8501
   - API Docs: http://localhost:8001/docs

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# API Service
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
BASE_URL=https://www.alphavantage.co/query
API_TIMEOUT=5

# Scraping Agent
USER_AGENT=Mozilla/5.0
SCRAPING_TIMEOUT=5

# Retriever Agent
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE_PATH=/app/data/vector_store

# Language Agent
GEMINI_API_KEY=your_gcp_service_key_here
GEMINI_MODEL=gemini-flash
TIMEOUT=10

# Streamlit
STREAMLIT_SERVER_PORT=8501
```

## 🚀 Services

| Service | Port | Description |
|---------|------|-------------|
| API Agent | 8001 | Financial data API endpoints |
| Scraping Agent | 8002 | Web scraping services |
| Retriever Agent | 8003 | Vector search and embeddings |
| Language Agent | 8005 | Text generation with Gemini |
| Orchestrator | 8004 | Agent coordination service |
| Streamlit UI | 8501 | Web interface |

## 📚 Documentation

For detailed documentation, please refer to [Documentation.md](Documentation.md).

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or support, please open an issue in the repository.

> "Today, your Asia tech allocation is 22% of AUM, up from 18% yesterday. TSMC beat estimates by 4%, Samsung missed by 2%. Regional sentiment is neutral with a cautionary tilt due to rising yields."

## System Architecture

### Core Components

1. **API Agent**
   - Fetches real-time and historical market data from financial APIs (AlphaVantage, Yahoo Finance)
   - Monitors market conditions and price movements

2. **Scraping Agent**
   - Automates data collection from financial filings and news sources
   - Implements efficient web scraping with Python loaders

3. **Retriever Agent**
   - Manages vector embeddings using FAISS or Pinecone
   - Implements semantic search for relevant financial information

4. **Analysis Agent**
   - Processes and interprets financial data
   - Identifies trends, anomalies, and key metrics

5. **Language Agent**
   - Generates human-like financial narratives
   - Utilizes advanced LLM capabilities for coherent reporting

6. **Voice Agent**
   - Handles speech-to-text (Whisper) and text-to-speech conversion
   - Manages voice I/O pipelines

### Technical Implementation

- **Backend**: FastAPI microservices for agent orchestration
- **AI/ML**: LangGraph and CrewAI for agent coordination
- **Data Storage**: Vector databases for efficient information retrieval
- **Deployment**: Containerized deployment with Docker
- **Frontend**: Interactive Streamlit dashboard

### Communication Flow

1. Voice input → Speech-to-Text conversion
2. Query processing by orchestrator
3. Data retrieval and analysis
4. Response generation via LLM
5. Text-to-Speech output or text display

### Error Handling

- Confidence-based fallback mechanism
- User clarification prompts when needed
- Graceful degradation of service

## Getting Started

### Prerequisites

- Python 3.11
- Docker (for containerized deployment)
- API keys for required services

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FinAI.git
cd FinAI

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Running the Application

```bash
# Start the services
docker-compose up -d

# Access the web interface
streamlit run streamlit_app/main.py
```


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the open-source community for their invaluable tools and libraries