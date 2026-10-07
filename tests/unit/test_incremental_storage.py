from uuid import uuid4

import pytest

from github_intelligence.indexing.storage import IncrementalVectorStore, publish_lexical
from github_intelligence.models import DocumentChunk
from github_intelligence.retrieval import BM25Retriever


def chunk():
    return DocumentChunk(
        chunk_id=str(uuid4()),
        document_id="1",
        chunk_index=0,
        text="useful source",
        title="Source",
        source_url="https://github.com/acme/repo/issues/1",
        source_type="issue",
        repository="acme/repo",
    )


def test_real_qdrant_upsert_delete_lookup_and_membership(tmp_path):
    store = IncrementalVectorStore(tmp_path / "vectors", "application")
    one, two = chunk(), chunk()
    try:
        with pytest.raises(RuntimeError, match="replacement"):
            store.replace([one], [[1, 0]], 2)
        with pytest.raises(RuntimeError, match="READY"):
            store.search([1, 0], top_k=10, repository=None)
        store.upsert([one, two], [[1, 0], [0, 1]], 2)
        store.upsert([one], [[1, 0]], 2)
        assert store.client.count("application", exact=True).count == 2
        assert set(store.lookup([one.chunk_id, two.chunk_id])) == {one.chunk_id, two.chunk_id}
        assert [m.chunk.chunk_id for m in store.search_membership([1, 0], [two.chunk_id], 10)] == [
            two.chunk_id
        ]
        store.delete([one.chunk_id])
        store.delete([one.chunk_id])
        assert set(store.lookup([one.chunk_id, two.chunk_id])) == {two.chunk_id}
        with pytest.raises(ValueError, match="dimension"):
            store.ensure(3)
    finally:
        store.client.close()


def test_lexical_atomic_failure_preserves_previous_and_formula(tmp_path, monkeypatch):
    from github_intelligence.indexing import storage

    path = tmp_path / "chunks.jsonl"
    one, two = chunk(), chunk()
    publish_lexical(path, [one])
    previous = path.read_bytes()

    def crash(*_):
        raise OSError("rename interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(storage.os, "replace", crash)
        with pytest.raises(OSError):
            publish_lexical(path, [two])
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob(".chunks-*"))
    publish_lexical(path, [one, two])
    original = BM25Retriever(tmp_path / "original.jsonl")
    original.index([one, two])
    assert BM25Retriever(path).search("useful") == original.search("useful")
