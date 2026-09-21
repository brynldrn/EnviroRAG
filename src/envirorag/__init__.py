import httpx
from pathlib import Path
from envirorag.retrieval import retrieve_documents

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"

def main() -> None:
    question = "What contamination was found at Northport?"
    documents = retrieve_documents(question)
    print(f"Retrieved {len(documents)} documents")

    for document in documents:
        print(document[:100])
        print("---")
        
    context = "\n\n---\n\n".join(documents)

    response = httpx.post(
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