# Development Guide

Welcome to the Patent Discovery System development guide! This document outlines the workflows, standards, and tools we use to build and maintain this project.

## 🛠️ Environment Setup

### 1. Global Tools
- **Docker & Docker Compose**: Essential for running the full stack locally.
- **Python 3.10+**: For backend development.
- **Node.js 18+**: For frontend development.
- **uv**: (Recommended) Fast Python package manager.

### 2. Environment Variables
Each app requires a `.env` file. We provide `.env.example` templates in each directory.
- `apps/api/.env.example`
- `apps/frontend/.env.example`

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
We use **Tailwind CSS 4** for styling. Please use utility classes and avoid custom CSS where possible.

## 🐳 Docker Workflow

For rapid development of the entire system, use Docker Compose:
```bash
docker-compose up

docker-compose up --build api
```

## 🧪 Testing

### Backend Tests
(To be implemented - contributions welcome!)
```bash
pytest
```

### Frontend Tests
(To be implemented - contributions welcome!)
```bash
npm test
```

## 🤝 Coding Standards

- **Python**: Follow PEP 8. Use type hints for all function signatures.
- **TypeScript**: Use strict mode. Avoid `any` whenever possible.
- **Git**: Use descriptive commit messages (following [Conventional Commits](https://www.conventionalcommits.org/)).

## 🚀 Deployment

The system is designed to be deployed via Docker containers. A sample `nginx.conf` is provided in the root to handle traffic between the frontend and backend.
