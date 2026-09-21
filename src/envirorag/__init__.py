import httpx
from pathlib import Path
from envirorag.retrieval import retrieve_documents
from envirorag.embeddings import embed, cosine_similarity

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"

def main() -> None:
    documents = retrieve_documents(
        "Where did we find TCE?",
        top_k=1,
    )

    print("\nRETRIEVED:")
    print(documents[0][:300])
    
    question = "What contamination was found at Northport?"
    documents = retrieve_documents(question)
    context = "\n\n---\n\n".join(documents)

    query = embed("Where did we find TCE?")

    candidate_a = embed(
        "Elevated trichloroethylene concentrations were detected "
        "in groundwater and soil."
    )

    candidate_b = embed(
        "The facility operated as a fuel storage depot."
    )

    candidate_c = embed(
        "Lead contamination was identified near the metal "
        "fabrication area."
    )

    print(
        "Trichloroethylene:",
        cosine_similarity(query, candidate_a),
    )

    print(
        "Fuel storage:",
        cosine_similarity(query, candidate_b),
    )

    print(
        "Lead contamination:",
        cosine_similarity(query, candidate_c),
    )

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