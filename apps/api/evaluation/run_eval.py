import asyncio

from app.services.rag.orchestrator import RAGOrchestrator
from evaluation.ragas_evaluator import load_dataset, run_ragas

test_queries = {

}

async def main():
    orchestrator = RAGOrchestrator.from_env()
    dataset = await load_dataset(orchestrator, test_queries)
    results = run_ragas(dataset)
    results.to_pandas().to_csv("ragas_results.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())
