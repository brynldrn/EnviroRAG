from pathlib import Path

from envirorag.embeddings import embed, cosine_similarity


DATA_DIR = Path("data")


def retrieve_documents(
    query: str,
    top_k: int = 1,
) -> list[str]:

    query_vector = embed(query)

    results: list[tuple[float, str]] = []

    for path in DATA_DIR.glob("*.txt"):
        content = path.read_text()
        document_vector = embed(content)

        score = cosine_similarity(query_vector, document_vector)

        print(f"{path.name}: {score}")

        results.append((score, content))
        
    results.sort(
        key=lambda result: result[0],
        reverse=True,
    )

    return [content for score, content in results[:top_k]]