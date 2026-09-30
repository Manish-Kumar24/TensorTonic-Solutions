def text_chunking(tokens: list, chunk_size: int, overlap: int) -> list:
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(tokens):
            break
    return chunks