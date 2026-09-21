import re
from pathlib import Path

DATA_DIR = Path("data")
STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "was",
    "were",
    "at",
    "in",
    "on",
    "what",
    "which",
    "where",
}

def tokenize(text: str) -> set[str]:
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return words - STOP_WORDS

def retrieve_documents(query: str, top_k: int = 1) -> list[str]:
    query_words = tokenize(query)
    results: list[tuple[int, str]] = []

    for path in DATA_DIR.glob("*.txt"):
        content = path.read_text()
        document_words = tokenize(content)

        matches = query_words & document_words

        score = len(matches)

        print(f"{path.name}: score={score}, matches={matches}")

        results.append((score, content))

    results.sort(key=lambda result: result[0], reverse=True)

    return [
        content 
        for score, content in results[:top_k] 
        if score > 0
    ]
