from app.rag.ingest import chunk_text


def test_chunk_text_basic():
    text = "a" * 1000
    chunks = chunk_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 500


def test_chunk_text_overlap():
    text = "a" * 600
    chunks = chunk_text(text)
    # with 500 chunk size and 50 overlap, 600 chars should give 2 chunks
    assert len(chunks) == 2


def test_chunk_text_empty():
    chunks = chunk_text("")
    assert chunks == []


def test_chunk_text_short():
    text = "Short text under chunk size."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text