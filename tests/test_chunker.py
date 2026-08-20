from app.rag.chunker import chunk_text


def test_chunk_text_keeps_overlap_metadata() -> None:
    chunks = chunk_text("abcdef" * 100, source="demo.txt", tenant_id="t1", chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].metadata["char_start"] == 0
    assert chunks[1].metadata["char_start"] == 80

