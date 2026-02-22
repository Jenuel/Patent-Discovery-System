# Patent Discovery System - Backend API

The backend API for the Patent Discovery System, providing AI-powered patent search, prior art discovery, and infringement analysis. It uses a robust Retrieval-Augmented Generation (RAG) architecture powered by Google's Gemini models, combined with sophisticated hierarchical retrieval methods across vector and sparse databases.

## 🌟 Key Features

* **Advanced RAG Orchestration**: Combines dense (Pinecone) and sparse (Elasticsearch) retrieval with optional hierarchical (Patent-level then Claim-level) fusion and re-ranking.
* **Dual-Index Pinecone Architecture**: Optimized for scale by optionally separating patent-level and claim-level vectors into distinct Pinecone vector indices.
* **Multiple Query Modes**:
  * `prior_art`: Comprehensive prior art search.
  * `infringement`: Detailed infringement-style element matching.
  * `landscape`: Technology landscape summaries and trend analysis.
* **Rich Filtering**: Filter searches by CPC codes, year ranges, and assignees.
* **Production-Ready FastAPI Server**: Includes CORS configuration, GZip compression, request logging, standard health/readiness probes, and robust error handling.

## 🏗️ Architecture

The backend consists of several key layers orchestrated by the `RAGOrchestrator`:

1. **API Routings (`app/api/v1/routes`)**: Defines REST endpoints using FastAPI. 
2. **Retrieval Strategy (`app/services/retrieval/`)**:
    * **Dense**: Embeddings retrieved from Pinecone.
    * **Sparse**: BM25 lexical search from Elasticsearch.
    * **Hierarchical**: Multi-stage retrieval traversing patent-level metadata to narrower claim-level data.
    * **Fusion**: Reciprocal Rank Fusion (RRF) to merge retrieve sources.
3. **Storage (`app/services/storage/`)**: Integrates with MongoDB to store and fetch raw text contexts / embedded values.
4. **LLM Integration (`app/services/llm/`)**: Google Gemini is used to synthesize evidence items into a high-quality, relevant answer.

## 🛠️ Technology Stack

* **Core**: Python 3.10+, FastAPI, Pydantic, Uvicorn
* **Database & Search**:
  * **Pinecone** (Vector Database for embeddings)
  * **Elasticsearch** (Lexical/BM25 Search)
  * **MongoDB** (Metadata and full-text document storage via `motor` async driver)
* **AI & LLM**: Google Generative AI (Gemini), OpenAI (Embeddings)
* **Package Management**: `uv` or `pip`

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed and set up:
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- Active instances for MongoDB, Elasticsearch, and Pinecone.
- API Keys for Google Gemini, OpenAI, and Pinecone.

### 1. Environment Configuration

Copy the example environment settings and populate the values for your stack:

```bash
cp .env.example .env
```

**Required Environment Variables (`.env`)**:

* **Server Settings**:
  * `ENV=dev` (or `prod`)
  * `CORS_ALLOW_ORIGINS=*` (comma-separated list for production)
* **Pinecone settings (Dual-Index mode)**:
  * `PINECONE_API_KEY=your-api-key`
  * `PINECONE_PATENT_INDEX=patents-patent-level-xxx...`
  * `PINECONE_CLAIM_INDEX=patents-claim-level-xxx...`
  * (Alternatively, for single index mode: `PINECONE_INDEX=your-index-host`)
* **AI Models**:
  * `OPENAI_API_KEY=your-openai-api-key`
  * `GEMINI_API_KEY=your-gemini-api-key`


### 2. Installation

We recommend using `uv` for fast dependency management and virtual environment creation.

```bash
# Create a virtual environment using uv
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

Alternatively, use standard `pip`:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Running the Application

To start the FastAPI server for local development, run Uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
*(The server will start at `http://127.0.0.0:8000/`)*

---

## 📚 API Reference

For full interactive API documentation, run the application and visit:
* **Interactive API Docs (Swagger)**: `http://localhost:8000/docs`
* **Alternative API Docs (ReDoc)**: `http://localhost:8000/redoc`

### Key Endpoints

#### 1. POST `/api/v1/query`
Main endpoint for querying the RAG system.

**Request Body** (`application/json`):
```json
{
  "query": "A machine learning algorithm for anomaly detection in cloud computing",
  "system_description": "We are building an anomaly detection engine using LSTM networks...",
  "filters": {
    "cpc_prefixes": ["G06N", "H04L"],
    "year_from": 2018,
    "year_to": 2024,
    "assignees": ["Google LLC", "Microsoft Corporation"]
  }
}
```

*Notes*: 
- `query` is required (minimum 3 characters). 
- If `system_description` is provided (or "infringement" keywords exist in the query), the mode dynamically switches to `infringement`.
- `filters` are optional but heavily recommended to target specific patent buckets.

**Response Structure**:
```json
{
  "mode": "prior_art",
  "answer": "Based on the retrieved patent documents, the use of LSTM networks for anomaly detection in cloud infrastructure is heavily documented...",
  "evidence": [
    {
      "chunk_id": "chunk-123",
      "patent_id": "US1234567B2",
      "level": "claim",
      "title": "Cloud Anomaly Detection",
      "claim_no": 1,
      "text": "A method comprising...",
      "score": 0.89,
      "source": "hybrid",
      "metadata": { ... }
    }
  ]
}
```

#### 2. System and Health endpoints

* `GET /`: Returns welcome message and basic API overview.
* `GET /health`: Basic operational health probe.
* `GET /ready`: Evaluates if all required API keys and services are configured and reachable.

---

## 📁 Project Structure

```text
apps/api/
├── app/
│   ├── api/          # Web routes and schemas (v1)
│   ├── core/         # Settings, environmental loading, logging configuration
│   ├── services/     # Core business logic domain
│   │   ├── indexing/ # Logic for uploading vectors and data
│   │   ├── llm/      # Google Gemini integration logic
│   │   ├── patents/  # Patent-specific parsing or utility
│   │   ├── rag/      # Complete RAG Orchestrator pipeline
│   │   ├── rerank/   # Re-ranking models integration (if active)
│   │   ├── retrieval/# Strategy pattern for Dense/Sparse/Hybrid/Hierarchical fetches
│   │   └── storage/  # Document/metadata layer interacting with MongoDB
│   └── main.py       # FastAPI application initialization
├── requirements.txt  # Dependencies definitions
└── README.md         # This file!
```