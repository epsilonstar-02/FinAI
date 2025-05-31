# FinAI - Advanced Multi-Agent Financial Intelligence System

**Version:** (Reflects overall project state, e.g., 1.0 after this refactor)

## Table of Contents

1.  [Overview](#overview)
2.  [Features](#features)
3.  [System Architecture](#system-architecture)
    *   [Core Agents](#core-agents)
    *   [User Interface](#user-interface)
    *   [Technology Stack](#technology-stack)
4.  [Project Structure](#project-structure)
5.  [Setup and Installation](#setup-and-installation)
    *   [Prerequisites](#prerequisites)
    *   [Environment Variables](#environment-variables)
    *   [Local Model Setup (Optional but Recommended)](#local-model-setup-optional-but-recommended)
    *   [Installation Steps](#installation-steps)
6.  [Running the System](#running-the-system)
    *   [Running Individual Agents](#running-individual-agents)
    *   [Running the Streamlit UI](#running-the-streamlit-ui)
    *   [Using Docker (Recommended for Deployment)](#using-docker-recommended-for-deployment)
7.  [Agent Details](#agent-details)
    *   [Orchestrator Agent](#orchestrator-agent)
    *   [API Agent](#api-agent)
    *   [Scraping Agent](#scraping-agent)
    *   [Retriever Agent](#retriever-agent)
    *   [Analysis Agent](#analysis-agent)
    *   [Language Agent](#language-agent)
    *   [Voice Agent](#voice-agent)
8.  [Streamlit User Interface](#streamlit-user-interface)
9.  [Configuration](#configuration)
10. [API Endpoints Summary](#api-endpoints-summary) (Brief overview of key agent endpoints)
11. [Contributing](#contributing)
12. [License](#license)

---

## 1. Overview

FinAI is an advanced financial intelligence system built using a multi-agent architecture. It aims to provide users with comprehensive financial insights by leveraging specialized agents for data retrieval, scraping, analysis, natural language processing, and voice interaction. The system can process user queries in text or voice, gather relevant financial data from various sources, perform analyses, and generate coherent, actionable briefings.

## 2. Features

*   **Multi-Agent Architecture:** Specialized agents for distinct tasks, promoting modularity and scalability.
*   **Multi-Provider Data Sources:** Integrates with multiple financial data APIs (e.g., Yahoo Finance, Alpha Vantage, FMP) and news sources with fallback mechanisms.
*   **Natural Language Interaction:** Supports queries via text and voice.
*   **Advanced Text Generation:** Utilizes multiple LLM providers (local Llama, HuggingFace, and optional commercial APIs like Gemini, OpenAI, Anthropic) with template-based prompting and caching.
*   **Voice Capabilities:** Speech-to-Text (STT) and Text-to-Speech (TTS) using various local and API-based providers.
*   **Data Scraping:** Capable of scraping news articles and SEC filings using robust extraction techniques.
*   **Vector Search & Retrieval:** Semantic search over ingested documents using a vector store (FAISS default).
*   **Financial Analysis:** Computes exposures, price changes, volatility, correlations, and various risk metrics.
*   **Interactive User Interface:** A Streamlit application provides a user-friendly way to interact with the system, configure settings, and view results.
*   **Configurable & Extensible:** Settings managed via environment variables and Pydantic models, designed for easy extension.
*   **Resilience:** Features retry mechanisms, provider fallbacks, and caching for improved reliability and performance.

## 3. System Architecture

FinAI comprises several microservices (agents) that communicate via HTTP APIs, orchestrated by a central Orchestrator Agent. A Streamlit application serves as the user interface.

### Core Agents:

*   **Orchestrator Agent:** The central coordinator. Receives user requests, determines the necessary steps, calls other agents, aggregates results, and formulates the final response.
*   **API Agent:** Fetches structured financial data (prices, historical data) from multiple financial data APIs (e.g., Yahoo Finance, Alpha Vantage, FMP).
*   **Scraping Agent:** Scrapes unstructured or semi-structured data from the web, such as news articles and SEC EDGAR filings.
*   **Retriever Agent:** Manages a vector store for documents. It ingests text data, creates embeddings, and allows for semantic similarity search to retrieve relevant context.
*   **Analysis Agent:** Performs quantitative financial analysis on provided data (e.g., price data, historical trends) to compute metrics like volatility, correlations, exposures, and risk assessments.
*   **Language Agent:** Leverages Large Language Models (LLMs) from multiple providers to generate natural language responses, summaries, and insights based on context provided by other agents.
*   **Voice Agent:** Handles Speech-to-Text (STT) and Text-to-Speech (TTS) functionalities, supporting various local and cloud-based voice processing providers.

### User Interface:

*   **Streamlit App:** Provides an interactive web interface for users to input queries (text/voice), configure agent parameters, view results (including charts and dashboards), and manage query history.

### Technology Stack:

*   **Backend (Agents):** Python, FastAPI, Pydantic, HTTPX, Tenacity
*   **Frontend (UI):** Streamlit
*   **LLMs:** LlamaCpp, HuggingFace Transformers, Google Gemini API, OpenAI API, Anthropic API (configurable)
*   **STT/TTS:** OpenAI Whisper (local/API), pyttsx3, gTTS, EdgeTTS, Google Cloud Speech-to-Text, Azure Speech, Vosk, Silero, Coqui, ElevenLabs, Amazon Polly (configurable)
*   **Data Scraping:** BeautifulSoup, Feedparser, Newspaper3k, Trafilatura, Readability-LXML, yfinance, secedgar
*   **Vector Store:** FAISS (default), ChromaDB (via Langchain integrations)
*   **Embeddings:** Sentence-Transformers (default), OpenAI Embeddings
*   **Data Analysis:** NumPy, Pandas
*   **Data Visualization (UI):** Plotly, Altair
*   **Audio (UI):** Streamlit-WebRTC, PyAV
*   **Configuration:** Python-dotenv, Pydantic-settings
*   **Caching:** Cachetools (in-memory for LLM), Diskcache (for Voice Agent)

## 4. Project Structure

```
finai-multi-agent-system/
├── agents/
│   ├── api_agent/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── config.py
│   │   ├── main.py
│   │   └── models.py
│   ├── analysis_agent/
│   ├── language_agent/
│   │   └── prompts/ (*.tpl files)
│   ├── retriever_agent/
│   ├── scraping_agent/
│   └── voice_agent/
├── orchestrator/
│   ├── __init__.py
│   ├── client.py
│   ├── config.py
│   ├── main.py
│   └── models.py
├── streamlit_app/
│   ├── __init__.py
│   ├── advanced_components.py
│   ├── app.py
│   ├── assets/ (e.g., logo.png)
│   ├── components.py
│   ├── requirements.txt (specific to streamlit app)
│   ├── static/ (if serving static assets directly via Streamlit)
│   ├── styles.css
│   └── utils.py
├── .env (template or actual, gitignored)
├── requirements.txt (main project requirements)
├── docker-compose.yml (Example for running all services)
└── README.md
```

## 5. Setup and Installation

### Prerequisites

*   Python 3.9+
*   `pip` for package installation
*   Access to a terminal or command prompt
*   (Optional but Recommended) Docker and Docker Compose for containerized deployment.
*   (Optional) API keys for commercial services you wish to use (OpenAI, Google Gemini, Alpha Vantage, FMP, ElevenLabs, etc.).
*   (Optional) Git for cloning the repository.

### Environment Variables

Create a `.env` file in the project root directory by copying `.env.example` (if provided) or creating a new one. Populate it with necessary configurations, especially API keys and local model paths.

**Example `.env` structure:**

```dotenv
# Orchestrator Configuration
ORCHESTRATOR_URL=http://localhost:8004
API_AGENT_URL=http://localhost:8001
SCRAPING_AGENT_URL=http://localhost:8002
RETRIEVER_AGENT_URL=http://localhost:8003
ANALYSIS_AGENT_URL=http://localhost:8007
LANGUAGE_AGENT_URL=http://localhost:8005
VOICE_AGENT_URL=http://localhost:8006

# Language Agent - LLM Keys (Optional - defaults to local/free if keys missing)
# GEMINI_API_KEY=your_gemini_api_key
# OPENAI_API_KEY=your_openai_api_key
# ANTHROPIC_API_KEY=your_anthropic_api_key
# HUGGINGFACE_API_KEY=your_huggingface_api_key # For HF Inference API

# Language Agent - Local Model Paths (if using local Llama/HF)
LLAMA_MODEL_PATH=./models/llama/your_llama_model.gguf # Example
# HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2 # Can be overridden for local HF

# API Agent - Financial Data Provider Keys (Optional - some providers are free)
# ALPHA_VANTAGE_KEY=your_alpha_vantage_key
# FMP_KEY=your_fmp_key # Free tier available

# Voice Agent - STT/TTS Provider Keys (Optional)
# WHISPER_API_KEY=your_openai_api_key # For Whisper API (if also used for LLM)
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/google_cloud_credentials.json
# AZURE_SPEECH_KEY=your_azure_speech_key
# AZURE_SPEECH_REGION=your_azure_speech_region
# ELEVENLABS_API_KEY=your_elevenlabs_key
# AWS_ACCESS_KEY_ID=your_aws_access_key
# AWS_SECRET_ACCESS_KEY=your_aws_secret_key
# AWS_REGION=us-east-1

# Voice Agent - Local Model Paths (if using local STT/TTS)
WHISPER_MODEL_PATH=./models/whisper/
VOSK_MODEL_PATH=./models/vosk/
# ... other local model paths for DeepSpeech, Silero, Coqui

# Retriever Agent
# OPENAI_API_KEY=your_openai_api_key # If using OpenAI embeddings

# Scraping Agent
SEC_API_KEY="YourCompanyName YourEmail@example.com" # For secedgar User-Agent

# General Log Level for Agents (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Streamlit App Environment (dev or prod) - for URL switching
STREAMLIT_ENV=dev
```

### Local Model Setup (Optional but Recommended)

For optimal performance and cost-effectiveness, especially with the Language, Voice, and Retriever agents, setting up local models is recommended.

1.  **LLMs (Language Agent):**
    *   **Llama:** Download a GGUF-formatted Llama model (e.g., from HuggingFace). Place it in a directory like `models/llama/` and update `LLAMA_MODEL_PATH` in your `.env` file. Ensure `llama-cpp-python` is installed (see `requirements.txt`).
    *   **HuggingFace Local:** Ensure `transformers` and `torch` (with CUDA/MPS if GPU is available) are installed. The `HUGGINGFACE_MODEL` setting in `config.py` (Language Agent) will be used. Models will be downloaded by the `transformers` library on first use to its cache.
2.  **STT Models (Voice Agent):**
    *   **Whisper (Local):** The `openai-whisper` library will download models to `WHISPER_MODEL_PATH` on first use. Ensure this path is writable.
    *   **Vosk:** Download a Vosk model for your desired language from [alphacephei.com](https://alphacephei.com/vosk/models) and extract it to the directory specified by `VOSK_MODEL_PATH`.
    *   **DeepSpeech:** Download DeepSpeech model files (.pbmm and .scorer) and place them in the `DEEPSPEECH_MODEL_PATH`.
3.  **TTS Models (Voice Agent):**
    *   **Silero/Coqui:** These libraries can often download models on first use if a specific model path isn't provided or valid. If using custom local models, set `SILERO_MODEL_PATH` or `COQUI_MODEL_PATH`.
4.  **Embedding Models (Retriever Agent):**
    *   **Sentence-Transformers:** The model specified by `EMBEDDING_MODEL` (e.g., `all-MiniLM-L6-v2`) will be downloaded by the `sentence-transformers` library on first use to its cache.

### Installation Steps

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd finai-multi-agent-system
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install main dependencies:**
    This `requirements.txt` should consolidate dependencies from all agents and the orchestrator.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` in the root should be a comprehensive list. If individual agent `requirements.txt` files exist, they might need to be merged or installed separately if developing agents in isolation.*

4.  **Install Streamlit app dependencies:**
    ```bash
    pip install -r streamlit_app/requirements.txt
    ```
5.  **Specific C++ Dependencies for some libraries (e.g., LlamaCpp, some audio libraries):**
    *   **LlamaCpp:** May require C++ compilation toolchain. Refer to `llama-cpp-python` documentation for OS-specific prerequisites (e.g., `CMake`, C++ compiler). Installation might look like:
        ```bash
        # Example with CMAKE arguments for GPU support (adjust as needed)
        # CMAKE_ARGS="-DLLAMA_CUBLAS=ON" pip install llama-cpp-python
        pip install llama-cpp-python
        ```
    *   **Audio Libraries:** Some audio processing libraries (like `pyaudio` for `speech_recognition` microphone input, or some STT/TTS backends) might have system-level dependencies (e.g., `portaudio` on Linux: `sudo apt-get install portaudio19-dev`). Check the respective library documentation. `streamlit-webrtc` handles browser-side audio capture.

## 6. Running the System

It's recommended to run each agent in a separate terminal or use a process manager like `pm2` or Docker Compose.

### Running Individual Agents

For each agent (Orchestrator, API, Scraping, Retriever, Analysis, Language, Voice):

1.  Navigate to the agent's directory, e.g., `cd orchestrator` or `cd agents/api_agent`.
2.  Ensure the `.env` file in the project root is configured. Agents load settings from it.
3.  Run the FastAPI application using Uvicorn:
    *   **Orchestrator:** `python -m uvicorn orchestrator.main:app --host 0.0.0.0 --port 8004 --reload`
    *   **API Agent:** `python -m uvicorn agents.api_agent.main:app --host 0.0.0.0 --port 8001 --reload`
    *   **Scraping Agent:** `python -m uvicorn agents.scraping_agent.main:app --host 0.0.0.0 --port 8002 --reload`
    *   **Retriever Agent:** `python -m uvicorn agents.retriever_agent.main:app --host 0.0.0.0 --port 8003 --reload`
    *   **Analysis Agent:** `python -m uvicorn agents.analysis_agent.main:app --host 0.0.0.0 --port 8007 --reload`
    *   **Language Agent:** `python -m uvicorn agents.language_agent.main:app --host 0.0.0.0 --port 8005 --reload`
    *   **Voice Agent:** `python -m uvicorn agents.voice_agent.main:app --host 0.0.0.0 --port 8006 --reload`

    *(Adjust ports based on your `.env` and agent `config.py` files if they differ from the defaults shown here, which are based on the Orchestrator's default URLs for them).*

### Running the Streamlit UI

1.  Navigate to the `streamlit_app` directory: `cd streamlit_app`
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
    The application will typically be available at `http://localhost:8501`.

### Using Docker (Recommended for Deployment)

A `docker-compose.yml` file would define services for each agent and the Streamlit app, simplifying deployment.

**Example `docker-compose.yml` structure (illustrative):**
```yaml
version: '3.8'
services:
  orchestrator:
    build: ./orchestrator
    ports:
      - "8004:8004"
    env_file: .env
    volumes: # Optional: for local model persistence if not baked into image
      - ./models:/app/models 
  
  api_agent:
    build: ./agents/api_agent
    ports:
      - "8001:8001"
    env_file: .env

  # ... Define services for other agents (scraping_agent, retriever_agent, etc.) ...

  language_agent:
    build: ./agents/language_agent
    ports:
      - "8005:8005"
    env_file: .env
    volumes: # Mount local models for Llama, Whisper, etc.
      - ./models:/app/models # Assuming models are stored in root/models

  voice_agent:
    build: ./agents/voice_agent
    ports:
      - "8006:8006"
    env_file: .env
    volumes:
      - ./models:/app/models
      - ./cache/voice_agent:/app/cache/voice_agent # Persist voice agent cache

  streamlit_app:
    build: ./streamlit_app
    ports:
      - "8501:8501"
    env_file: .env # Ensure Streamlit app can read ORCH_URL etc.
    # Depends on other services if direct calls were made, but primarily talks to Orchestrator.
    depends_on:
      - orchestrator # Example dependency
```
Each service would need a `Dockerfile`.

**To run with Docker Compose:**
```bash
docker-compose up --build
```

## 7. Agent Details

### Orchestrator Agent
*   **Port (default):** 8004
*   **Purpose:** Central coordinator of the multi-agent system.
*   **Key Endpoint:** `POST /run`
    *   Accepts `RunRequest` (input query, mode, parameters).
    *   Dynamically calls other agents based on request.
    *   Aggregates information and uses Language Agent for final response generation.
    *   Handles STT/TTS internally if `mode="voice"` and audio/text is provided.
*   **Client (`orchestrator/client.py`):** `AgentClient` used by orchestrator to call downstream agents, featuring retries.

### API Agent
*   **Port (default):** 8001
*   **Purpose:** Fetches structured financial market data.
*   **Providers:** Yahoo Finance, Alpha Vantage, Financial Modeling Prep (FMP), with fallback.
*   **Key Endpoints:**
    *   `GET /price?symbol=...[&provider=...]`: Current price.
    *   `GET /historical?symbol=...&start_date=...&end_date=...[&provider=...]`: Historical OHLCV data.
    *   `POST /multi-price`: Fetches price for a symbol from multiple specified providers.
    *   `GET /compare-providers/{symbol}`: Compares prices from all available providers for a symbol.
*   **Features:** Provider fallback, retries, configurable provider priority.

### Scraping Agent
*   **Port (default):** 8002
*   **Purpose:** Scrapes web content like news and SEC filings.
*   **Methods:** Uses `feedparser`, `newspaper3k`, `trafilatura`, `readability-lxml`, `BeautifulSoup` for robust content extraction. `yfinance` for some company news/data, `secedgar` for SEC filings.
*   **Key Endpoints:**
    *   `GET /news?topic=...[&limit=...&source=...]`: General news.
    *   `GET /company/news/{symbol}[?limit=...]`: Company-specific news.
    *   `GET /market/news[?limit=...&category=...]`: General market news.
    *   `POST /filing/by-url`: Processes a single SEC filing from its URL.
    *   `GET /company/filings/{symbol}[?form_type=...&limit=...]`: Company SEC filings.
    *   `GET /company/profile/{symbol}`: Company profile.
    *   `GET /company/earnings/{symbol}`: Company earnings data.
*   **Features:** Multi-method content extraction, retry on extraction.

### Retriever Agent
*   **Port (default):** 8003
*   **Purpose:** Manages document ingestion, embedding, and semantic retrieval using a vector store.
*   **Vector Store Backend (default):** FAISS (via Langchain). Configurable for ChromaDB, etc.
*   **Embeddings (default):** Sentence-Transformers (`all-MiniLM-L6-v2`). Configurable for OpenAI.
*   **Key Endpoints:**
    *   `POST /ingest`: Ingests a list of documents into a specified namespace.
    *   `POST /retrieve`: Retrieves documents similar to a query, with filtering and top-k.
    *   `DELETE /documents`: Deletes documents by ID or clears a namespace.
    *   `PUT /documents/{document_id}`: Updates a document.
    *   `GET /stats[?namespace=...]`: Vector store statistics.
    *   `GET /namespaces`: Lists available namespaces.
*   **Features:** Namespaced document storage, robust embedding, Langchain integration.

### Analysis Agent
*   **Port (default):** 8007
*   **Purpose:** Performs financial calculations and analysis.
*   **Providers:** "default" (basic calculations) and "advanced" (more sophisticated metrics).
*   **Key Endpoint:** `POST /analyze`
    *   Accepts current prices and historical data.
    *   Calculates exposures, price changes, volatility.
    *   Optionally computes correlations and risk metrics (Sharpe, Max Drawdown, VaR, Beta, Sortino, Calmar, CVaR).
    *   Generates a textual summary with alerts.
*   **Features:** Provider fallback, caching, rate limiting, configurable analysis parameters.

### Language Agent
*   **Port (default):** 8005
*   **Purpose:** Generates natural language text using LLMs.
*   **LLM Providers:** Llama (local), HuggingFace (local/API), Google Gemini, OpenAI, Anthropic, with fallback.
*   **Key Endpoint:** `POST /generate`
    *   Accepts a query, context data, and optional template name.
    *   Renders a Jinja2 prompt template.
    *   Calls the selected/default LLM provider.
    *   Allows overriding temperature and max_tokens per request.
*   **Features:** Multi-LLM support, template-based prompting, response caching, retries.

### Voice Agent
*   **Port (default):** 8006
*   **Purpose:** Provides Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities.
*   **STT Providers:** Whisper (local/API), Google Cloud Speech, Azure Speech, Vosk, DeepSpeech, SpeechRecognition library.
*   **TTS Providers:** pyttsx3 (offline), gTTS, EdgeTTS, ElevenLabs, Amazon Polly, Silero, Coqui.
*   **Key Endpoints:**
    *   `POST /stt`: Converts uploaded audio file to text.
    *   `POST /tts`: Converts text to speech audio (MP3 stream or Base64).
    *   `GET /voices[?provider=...]`: Lists available TTS voices (placeholder, can be extended).
*   **Features:** Multi-provider STT/TTS, audio preprocessing (denoising, VAD), caching, fallback.

## 8. Streamlit User Interface

*   **Access:** Typically `http://localhost:8501`
*   **Functionality:**
    *   **Market Overview Tab:** Displays sample stock information and market trend charts.
    *   **Portfolio Analysis Tab:** Renders detailed analysis dashboards based on data from the Analysis Agent. Includes a button to generate sample analysis.
    *   **Financial Assistant Tab:** Main interaction point.
        *   Supports text and voice input modes.
        *   Voice input uses browser microphone via `streamlit-webrtc`. Recorded audio is sent to the Orchestrator.
        *   Users can input queries, and the system generates a financial brief.
    *   **Sidebar:**
        *   Configure LLM provider, model, temperature, max tokens.
        *   Configure STT/TTS providers, voice, speaking rate, pitch.
        *   Configure Analysis Agent provider and options (correlations, risk metrics).
        *   Set context parameters: ticker symbols for data fetching, news scraping limits, document retrieval K-value.
        *   View session information and clear chat history.
        *   Theme toggle (light/dark).
*   **Interaction Flow:**
    1.  User configures settings in the sidebar.
    2.  User inputs query (text or voice) in the "Financial Assistant" tab.
    3.  Streamlit app constructs a request for the Orchestrator Agent.
        *   For voice mode, raw recorded audio (base64 encoded) is included in the parameters sent to the Orchestrator. The Orchestrator handles STT.
        *   All selected configurations and contextual parameters are packaged into `RunRequest.params`.
    4.  Orchestrator processes the request, interacts with other agents.
    5.  Streamlit app displays the Orchestrator's response:
        *   Generated text brief.
        *   If voice mode, plays back TTS audio received from Orchestrator.
        *   Shows agent processing steps, raw JSON response, and any errors.
        *   Provides options to download the brief (PDF, Text, Audio).
    6.  Interaction is added to query history.

## 9. Configuration

*   **Primary Configuration:** `.env` file in the project root. See [Environment Variables](#environment-variables) section for details.
*   **Agent-Specific Configs:** Each agent (and the orchestrator) has a `config.py` file defining Pydantic `Settings` models. These models load values from the `.env` file and provide type validation and defaults.
*   **Streamlit UI Configuration:** The UI allows users to override many default agent parameters for a specific query (e.g., LLM model, analysis options). These selected parameters are then passed to the Orchestrator.

## 10. API Endpoints Summary

*   **Orchestrator (`/orchestrator`):**
    *   `POST /run`: Main entry point for user queries.
*   **API Agent (`/api_agent`):**
    *   `GET /price`: Get current stock price.
    *   `GET /historical`: Get historical stock data.
*   **Scraping Agent (`/scraping_agent`):**
    *   `GET /news`: Get news by topic.
    *   `GET /company/news/{symbol}`: Company-specific news.
    *   `POST /filing/by-url`: Get SEC filing by URL.
*   **Retriever Agent (`/retriever_agent`):**
    *   `POST /ingest`: Add documents to vector store.
    *   `POST /retrieve`: Search documents.
*   **Analysis Agent (`/analysis_agent`):**
    *   `POST /analyze`: Perform financial analysis.
*   **Language Agent (`/language_agent`):**
    *   `POST /generate`: Generate text using LLM.
*   **Voice Agent (`/voice_agent`):**
    *   `POST /stt`: Speech-to-Text.
    *   `POST /tts`: Text-to-Speech.

Each agent also has a `GET /health` endpoint. Refer to individual agent `main.py` files or OpenAPI docs (`/docs` on each agent's port) for detailed request/response schemas.

## 11. Contributing

(Placeholder for contribution guidelines if this were an open project)
*   Fork the repository.
*   Create a new branch for your feature or bug fix.
*   Make your changes.
*   Ensure code is formatted (e.g., using Black) and linted (e.g., using Flake8).
*   Add/update tests for your changes.
*   Submit a pull request.

## 12. License

This project is licensed under the Apache 2.0 License - see the LICENSE.md file for details.