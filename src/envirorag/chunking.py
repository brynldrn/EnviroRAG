def chunk_text(
    text: str,
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[str]:
    if chunk_size <= overlap or chunk_size <= 0 or overlap < 0:
        raise ValueError("chunk_size must be greater than overlap")

    words: list[str] = text.split()

    # How far should the window move each iteration?
    step: int = chunk_size - overlap

    chunks: list[str] = []

    for start in range(0, len(words), step):
        chunk_words: list[str] = words[start:start + chunk_size]

        # check if chunk is smaller than the chunk size, if so, break the loop and carry over the remaining words to the previous chunk without duplicating the last overlap words
        if len(chunk_words) < chunk_size and chunks:
            remaining_words: list[str] = chunk_words[overlap:]
            chunks[-1] += " " + " ".join(remaining_words)
            break

        chunk: str = " ".join(chunk_words)

        chunks.append(chunk)

    return chunks