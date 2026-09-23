from envirorag.models import DocumentChunk

def chunk_text(
    text: str,
    source: str = "",
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[DocumentChunk]:
    if chunk_size <= overlap or chunk_size <= 0 or overlap < 0:
        raise ValueError("chunk_size must be greater than overlap")

    words: list[str] = text.split()

    # How far should the window move each iteration?
    step: int = chunk_size - overlap

    chunks: list[DocumentChunk] = []

    print(chunks)

    for start in range(0, len(words), step):
        chunk_words: list[str] = words[start:start + chunk_size]

        # check if chunk is smaller than the chunk size, if so, break the loop and carry over the remaining words to the previous chunk without duplicating the last overlap words
        if len(chunk_words) < chunk_size and chunks:
            remaining_words: list[str] = chunk_words[overlap:]
            remaining_chunk: str = " " + " ".join(remaining_words)
            chunks[-1].text += remaining_chunk
            break

        chunks.append(
            DocumentChunk(
                text=" ".join(chunk_words),
                source=source,
                chunk_index=len(chunks),
            )
        )

    return chunks