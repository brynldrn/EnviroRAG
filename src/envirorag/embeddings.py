import httpx
import math

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "qwen3-embedding:0.6b"

def embed(text: str) -> list[float]:
    payload = {
            "model": EMBED_MODEL,
            "input": text
        }

    response = httpx.post(
        OLLAMA_EMBED_URL,
        json=payload,
        timeout=120.0
    )

    response.raise_for_status()
    data = response.json()
    embeddings = data["embeddings"]

    if not embeddings:
        raise RuntimeError(
            f"Ollama returned no embeddings: {data}"
        )

    return embeddings[0]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    return dot_product / (magnitude_a * magnitude_b)