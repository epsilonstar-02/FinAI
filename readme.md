# FinAI: Intelligent Financial Assistant

## Overview
FinAI is an advanced, multi-agent financial intelligence system designed to provide real-time market insights through natural language interaction. The system delivers spoken market briefings via an intuitive Streamlit interface, powered by a sophisticated backend of specialized AI agents.

## Key Features

- **Real-time Market Analysis**: Get up-to-the-minute financial data and insights
- **Voice-Activated Interface**: Natural language interaction through speech-to-text and text-to-speech
- **Multi-Agent Architecture**: Specialized agents working in harmony to process and analyze financial data
- **Retrieval-Augmented Generation (RAG)**: Combines real-time data with contextual knowledge
- **Open Source**: Built with transparency and community collaboration in mind

## Use Case: Morning Market Briefing

Imagine starting your trading day with a simple voice command:

> "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"

FinAI responds with a concise, informative briefing:

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