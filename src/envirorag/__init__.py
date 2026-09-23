from httpx._models import Response
import httpx
from pathlib import Path
from envirorag.retrieval import retrieve_documents
# from envirorag.embeddings import embed, cosine_similarity

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"

def main() -> None:
    documents: list[str] = retrieve_documents(
        "Where did we find TCE?",
        top_k=3,
    )

    for i, document in enumerate(documents, start=1):
        print(f"\n--- RESULT {i} ---")
        print(document)
    
    question = "What contamination was found at Northport?"
    documents: list[str] = retrieve_documents(question)
    context: str = "\n\n---\n\n".join(documents)

    return

    response: Response = httpx.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": """
                    You are EnviroRAG, an internal environmental
                    engineering assistant.

                    Answer questions using the supplied company records.

                    If the records do not contain enough information
                    to answer the question, say so. Do not invent facts.
                    """,
                },
                {
                    "role": "user",
                    "content": f"""
                    COMPANY RECORD:

                    {context}

                    QUESTION:

                    {question}
                    """,
                },
            ],
            "stream": False
        },
        timeout=120.0
    )

    response.raise_for_status()
    data = response.json()
    print(data["message"]["content"])