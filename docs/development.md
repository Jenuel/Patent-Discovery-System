# Development Guide

Welcome to the Patent Discovery System development guide! This document outlines the workflows, standards, and tools we use to build and maintain this project.

## 🛠️ Environment Setup

### 1. Global Tools
- **Docker & Docker Compose**: Essential for running the full stack locally.
- **Python 3.10+**: For backend development.
- **Node.js 18+**: For frontend development.
- **uv**: (Recommended) Fast Python package manager.

### 2. Environment Variables
- `apps/api/.env` — `GEMINI_API_KEY`, `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `MONGODB_URI`, plus the optional retrieval-tuning vars documented in [`apps/api/README.md`](../apps/api/README.md).
- `apps/frontend/.env` — `VITE_API_URL`, only needed when not proxying to `http://localhost:8000`.

## 📡 Backend Development (FastAPI)

### Setup
```bash
cd apps/api
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Running Locally
```bash
uvicorn app.main:app --reload
```

### Key Modules
- `app/services/rag/orchestrator.py`: The heart of the retrieval-generation pipeline.
- `app/services/retrieval/`: Contains dense, sparse, and hierarchical retrieval logic.
- `app/api/v1/routes/query.py`: Main API endpoint for patent queries.

## 🎨 Frontend Development (React)

### Setup
```bash
cd apps/frontend
npm install
```

### Running Locally
```bash
npm run dev
```

### Styling Standards
No Tailwind — the frontend uses a hand-written CSS design system under `src/styles/` (`design-system.css` for tokens/primitives, `theme.css` for the `:root` overrides, `app.css` for the two screens). See [`apps/frontend/README.md`](../apps/frontend/README.md) for the layering and the structural rules (0px radius, 2px dividers, etc).

## 🐳 Docker Workflow

For rapid development of the entire system, use Docker Compose:
```bash
docker-compose up

docker-compose up --build api
```

## 🧪 Testing

### Backend Tests
`apps/api/tests/` covers fusion, hierarchical retrieval, the reranker, evidence formatting, the Qdrant store, and the eval harness itself.
```bash
cd apps/api
pytest
```

Retrieval quality (MRR/recall/nDCG ablations across the fusion arms) lives separately in `apps/api/evaluation/` — see [`docs/evaluation.md`](./evaluation.md).

### Frontend Tests
Not implemented yet — no `test` script in `apps/frontend/package.json`. Contributions welcome.

## 🤝 Coding Standards

- **Python**: Follow PEP 8. Use type hints for all function signatures.
- **TypeScript**: Use strict mode. Avoid `any` whenever possible.
- **Git**: Use descriptive commit messages (following [Conventional Commits](https://www.conventionalcommits.org/)).

## 🚀 Deployment

The system is designed to be deployed via Docker containers, orchestrated by the root `docker-compose.yml`. The frontend container serves its build with nginx, configured by `apps/frontend/nginx.conf`.
