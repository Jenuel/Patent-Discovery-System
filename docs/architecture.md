# Architecture Deep Dive: Patent Discovery RAG

This document provides a detailed overview of the architectural decisions and technical implementations of the Patent Discovery System.

## 🧠 Core Philosophy

Searching for patents is different from standard web search. Patents have a hierarchical structure (Patent -> Claims -> Elements) and require high recall to ensure no prior art is missed. Our architecture is designed to address these specific needs through a multi-stage RAG pipeline.

## 🚀 The RAG Pipeline

The `RAGOrchestrator` in the backend manages the end-to-end flow of a query:

### 1. Query Processing & Embedding
When a user submits a query, the system first determines the search mode (Prior Art, Infringement, or Landscape). It then uses the **OpenAI `text-embedding-3-small`** model to convert the query into a high-dimensional vector.

### 2. Hierarchical Retrieval Strategy
To optimize for both scale and relevance, we employ a **Dual-Index Hierarchical Search**:

- **Dense Search (Pinecone)**: Performs semantic search across patent-level embeddings to find the most relevant patent documents.
- **Sparse Search (Elasticsearch)**: Uses BM25 lexical search to catch specific technical terms and patent numbers that semantic search might miss.
- **Hierarchical Expansion**: Once candidate patents are identified, the system drills down into the **Claim-level Index** to find specific passages (claims) that match the query requirements.

### 3. Reciprocal Rank Fusion (RRF)
We merge results from different sources (Dense, Sparse, Patent-level, and Claim-level) using the **Reciprocal Rank Fusion (RRF)** algorithm. This ensures that items appearing at the top of multiple retrieval strategies are prioritized.

### 4. Knowledge Fetching (MongoDB)
Since vector databases are optimized for search but not for large text storage, the full text of patents and claims is stored in **MongoDB**. The orchestrator fetches the raw text for the top-N candidates identified in the previous step.

### 5. LLM Synthesis (Google Gemini)
The retrieved evidence, along with the original query and system instructions, is passed to **Google Gemini 1.5 Pro**. The LLM performs:
- **Evidence Extraction**: Identifying exactly which part of a patent claim is relevant.
- **Synthesis**: Answering the user's query in natural language, citing specific patent IDs.
- **Reasoning**: Explaining *why* a particular patent is considered prior art or potentially infringing.

## 📊 Data Model

### Patent Document (MongoDB)
```json
{
  "patent_id": "US1234567B2",
  "title": "Quantum Computing Architecture",
  "abstract": "...",
  "assignee": "Quantum Corp",
  "cpc_codes": ["G06N10/00"],
  "publication_date": "2023-01-01",
  "claims": [
    {"number": 1, "text": "...", "is_independent": true},
    {"number": 2, "text": "...", "is_independent": false}
  ]
}
```

### Vector Index (Pinecone)
- **Namespace: `patents`**: Embeddings of patent abstracts and titles.
- **Namespace: `claims`**: Embeddings of individual patent claims.

## 🛠️ Design Decisions & Rationale

| Decision | Rationale |
| :--- | :--- |
| **Why FastAPI?** | High performance with async/await support, perfect for IO-bound RAG tasks. |
| **Why Pinecone + Elasticsearch?** | Hybrid search (Dense + Sparse) is the gold standard for high-recall retrieval. |
| **Why Gemini 1.5 Pro?** | Largest context window for processing long patent documents and superior reasoning capabilities. |
| **Why Nginx as a Reverse Proxy?** | Handles TLS termination, GZip compression, and static file serving for the React app. |

## 🔜 Future Enhancements
- **Multi-Modal Retrieval**: Supporting search through patent figures and diagrams.
- **Reranking Layer**: Adding a cross-encoder (e.g., Cohere or Gemini-based) to further refine retrieval results.
- **Citation Graph Search**: Using patent citations as additional signals for relevance.
