from envirorag.models import DocumentChunk
from pathlib import Path

from envirorag.embeddings import embed, cosine_similarity
from envirorag.chunking import chunk_text


DATA_DIR = Path("data")


def retrieve_documents(
    query: str,
    top_k: int = 1,
) -> list[DocumentChunk]:

    query_vector: list[float] = embed(query)

    results: list[tuple[float, DocumentChunk]] = []

    for path in DATA_DIR.glob("*.txt"):
        content: str = path.read_text()
        chunks: list[DocumentChunk] = chunk_text(content, path.name)

        for chunk in chunks:
            chunk_vector: list[float] = embed(chunk.text)

            score: float = cosine_similarity(query_vector, chunk_vector)

            results.append((score, chunk))
        
    results.sort(
        key=lambda result: result[0],
        reverse=True,
    )

    return [chunk for score, chunk in results[:top_k]]