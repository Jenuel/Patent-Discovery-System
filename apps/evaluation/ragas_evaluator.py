from datasets import Dataset

from app.services.rag.orchestrator import RAGOrchestrator


async def load_dataset(
    orchestrator: RAGOrchestrator,
    questions: list[str],
    ground_truths: list[list[str]] | None = None,
) -> Dataset:
    samples = {"question": [], "answer": [], "contexts": []}
    if ground_truths:
        samples["ground_truths"] = ground_truths

    for i, question in enumerate(questions):
        response = await orchestrator.run(question)
        samples["question"].append(question)
        samples["answer"].append(response["answer"])
        samples["contexts"].append([e.text for e in response.evidence if e.text])
    
    return Dataset.from_dict(samples)
