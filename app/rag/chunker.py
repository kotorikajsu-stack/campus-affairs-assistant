from app.rag.schemas import DocumentChunk


def chunk_text(
    text: str,
    *,
    source: str,
    tenant_id: str,
    chunk_size: int = 420,
    overlap: int = 60,
) -> list[DocumentChunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    index = 0
    step = chunk_size - overlap

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{source}:{index}",
                    text=chunk,
                    source=source,
                    tenant_id=tenant_id,
                    metadata={"char_start": start, "char_end": end},
                )
            )
        if end == len(normalized):
            break
        start += step
        index += 1

    return chunks

