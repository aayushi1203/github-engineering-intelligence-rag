"""Coordinator behavior against real PostgreSQL and embedded Qdrant, no mocked stores."""

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

import pytest

pytest.importorskip("sqlalchemy")
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from github_intelligence.github.scans import ScanStatus
from github_intelligence.indexing.coordinator import IncrementalCoordinator, IndexingError
from github_intelligence.indexing.models import ChunkManifest, IndexOperation
from github_intelligence.indexing.policy import chunk_identity
from github_intelligence.indexing.storage import IncrementalVectorStore
from github_intelligence.ingestion.models import IngestionReport, RepositoryMetadata, SourceScan
from github_intelligence.ingestion.snapshot import report_path
from github_intelligence.models import GitHubDocument
from github_intelligence.persistence.identity import source_key
from github_intelligence.persistence.models import Document
from github_intelligence.retrieval import DocumentChunker


class CountingEncoder:
    dimension = 3

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, float("changed" in text), float("other" in text)] for text in texts]


class CountingChunker(DocumentChunker):
    def __init__(self):
        super().__init__(max_words=20, overlap_words=0)
        self.calls = 0

    def chunk(self, document):
        self.calls += 1
        return super().chunk(document)


def doc(key="1", body="useful source", family="issue", repository="acme/repo"):
    metadata = {"repository_github_id": "100" if repository == "acme/repo" else "200"}
    if family == "documentation":
        metadata.update(path=key, branch="main", commit_sha="a" * 40, blob_sha="b" * 40)
    value = GitHubDocument(
        repository=repository,
        document_type=family,
        github_id=key,
        title="Source",
        text=body,
        source_url=f"https://github.com/{repository}/issues/1",
        metadata=metadata,
    )
    value.metadata["source_key"] = source_key(value)
    return value


def snapshot(
    tmp_path,
    documents,
    *,
    name="acme/repo",
    statuses=None,
    requested=None,
    docs_branch="main",
    extra_observed=None,
    head_sha="a" * 40,
    tree_sha="c" * 40,
):
    path = tmp_path / (uuid4().hex + ".jsonl")
    payload = "".join(d.model_dump_json() + "\n" for d in documents).encode()
    path.write_bytes(payload)
    families = ["issue", "documentation", "pull_request", "issue_comment"]
    sources = {family: SourceScan(status=ScanStatus.COMPLETE) for family in families}
    sources["issue"].scope = {"state": "all"}
    sources["documentation"].scope = {
        "branch": docs_branch,
        "head_sha": head_sha,
        "tree_sha": tree_sha,
    }
    for d in documents:
        family = d.document_type.value
        sources.setdefault(family, SourceScan(status=ScanStatus.COMPLETE))
        sources[family].documents_emitted += 1
        sources[family].observed_source_keys.append(source_key(d))
    for family, state in (statuses or {}).items():
        sources[family].status = state
    for family, keys in (extra_observed or {}).items():
        sources[family].observed_source_keys.extend(keys)
    report = IngestionReport(
        repository=RepositoryMetadata(
            github_id="100" if name == "acme/repo" else "200",
            requested_name=name,
            name=name,
            github_url=f"https://github.com/{name}",
            default_branch=docs_branch,
            visibility="public",
            head_sha=head_sha,
            tree_sha=tree_sha,
        ),
        requested_families=requested if requested is not None else list(sources),
        sources=sources,
        finished_at=datetime.now(UTC),
        snapshot_sha256=hashlib.sha256(payload).hexdigest(),
    )
    report_path(path).write_text(report.model_dump_json())
    return path


@pytest.fixture
def application(migrated_engine, tmp_path):
    store = IncrementalVectorStore(tmp_path / "qdrant", "application")
    encoder, chunker = CountingEncoder(), CountingChunker()
    coordinator = IncrementalCoordinator(
        migrated_engine,
        store,
        encoder,
        tmp_path / "lexical",
        embedding_version="test-v1",
        chunker=chunker,
    )
    try:
        yield coordinator, encoder, chunker
    finally:
        coordinator.store.client.close()


def test_new_unchanged_changed_shrinkage_and_selective_embedding(application, tmp_path):
    coordinator, encoder, chunker = application
    first = coordinator.run(
        snapshot(tmp_path, [doc(body=" ".join(["word"] * 45)), doc("2")]), repository="acme/repo"
    )
    assert first.status == "READY" and first.plan["counts"]["new"] == 2
    pinned = coordinator.ready("acme/repo")
    calls, embeddings = chunker.calls, len(encoder.calls)
    second_input = snapshot(tmp_path, [doc(body=" ".join(["word"] * 45)), doc("2")])
    second = coordinator.run(second_input, repository="acme/repo")
    assert second.plan["counts"]["unchanged"] == 2
    assert chunker.calls == calls and len(encoder.calls) == embeddings
    assert set(coordinator.ready("acme/repo").ids) == set(pinned.ids)
    replay = coordinator.run(second_input, repository="acme/repo")
    assert replay.id == second.id and replay.attempts == 1
    changed = coordinator.run(
        snapshot(tmp_path, [doc(body="changed content"), doc("2")]), repository="acme/repo"
    )
    assert changed.plan["counts"]["changed"] == 1
    assert chunker.calls == calls + 1
    assert encoder.calls[-1] == ["Source\n\nchanged content"]
    current = coordinator.ready("acme/repo")
    assert len(current.ids) == 2 and set(current.ids) != set(pinned.ids)
    assert len(current.vector_search([1, 0, 0])) == 2
    # Old pinned revision remains independently usable after publication too.
    assert len(pinned.vector_search([1, 0, 0])) == len(pinned.ids)


def test_metadata_only_changes_reuse_embeddings(application, tmp_path):
    coordinator, encoder, _ = application
    coordinator.run(snapshot(tmp_path, [doc()]), repository="acme/repo")
    changed = doc()
    changed.metadata["author"] = "another"
    previous_calls = len(encoder.calls)
    coordinator.run(snapshot(tmp_path, [changed]), repository="acme/repo")
    assert len(encoder.calls) == previous_calls
    assert (
        coordinator.ready("acme/repo").vector_search([1, 0, 0])[0].chunk.metadata["author"]
        == "another"
    )


@pytest.mark.parametrize(
    "status", [ScanStatus.LIMITED, ScanStatus.TRUNCATED, ScanStatus.FAILED, ScanStatus.NOT_STARTED]
)
def test_incomplete_scan_refuses_deletion(application, tmp_path, status):
    coordinator, _, _ = application
    coordinator.run(
        snapshot(tmp_path, [doc("README.md", family="documentation")]), repository="acme/repo"
    )
    result = coordinator.run(
        snapshot(tmp_path, [], statuses={"documentation": status}),
        repository="acme/repo",
        reconcile=True,
    )
    assert result.plan["counts"]["deleted"] == 0 and result.plan["counts"]["skipped"] == 1
    assert len(coordinator.ready("acme/repo").ids) == 1


@pytest.mark.parametrize("reason", ["excluded", "branch", "mutable", "observed", "disabled"])
def test_ambiguous_or_out_of_scope_absence_preserved(application, tmp_path, reason):
    coordinator, _, _ = application
    original = doc() if reason == "mutable" else doc("README.md", family="documentation")
    coordinator.run(snapshot(tmp_path, [original]), repository="acme/repo")
    kwargs = {}
    if reason == "excluded":
        kwargs["requested"] = ["issue"]
    if reason == "branch":
        kwargs["docs_branch"] = "other"
    if reason == "observed":
        kwargs["extra_observed"] = {"documentation": [source_key(original)]}
    result = coordinator.run(
        snapshot(tmp_path, [], **kwargs), repository="acme/repo", reconcile=reason != "disabled"
    )
    assert result.plan["counts"]["deleted"] == 0 and result.plan["preserved"]


def test_safe_documentation_deletion_and_empty_ready(application, tmp_path):
    coordinator, _, _ = application
    coordinator.run(
        snapshot(tmp_path, [doc("README.md", family="documentation")]), repository="acme/repo"
    )
    old = coordinator.ready("acme/repo")
    result = coordinator.run(
        snapshot(tmp_path, [], head_sha="d" * 40, tree_sha="e" * 40),
        repository="acme/repo",
        reconcile=True,
    )
    assert result.plan["counts"]["deleted"] == 1
    current = coordinator.ready("acme/repo")
    assert current.ids == [] and current.lexical.search("source") == []
    assert current.vector_search([1, 0, 0]) == []
    assert old.vector_search([1, 0, 0])
    with Session(coordinator.engine) as session:
        assert session.scalars(select(Document)).one().index_status == "deleted"


@pytest.mark.parametrize(
    "stage", ["prepared", "vectors_written", "lexical_built", "verified", "publishing"]
)
def test_partial_failure_replay_preserves_ready(application, tmp_path, stage):
    coordinator, encoder, _ = application
    original = coordinator.run(snapshot(tmp_path, [doc()]), repository="acme/repo")
    second = snapshot(tmp_path, [doc(body="changed")])

    def crash(boundary):
        if boundary == stage:
            raise RuntimeError("simulated interruption")

    coordinator.boundary = crash
    with pytest.raises(IndexingError):
        coordinator.run(second, repository="acme/repo")
    ready = coordinator.ready("acme/repo")
    assert ready.revision == original.id and ready.vector_search([1, 0, 0])
    with Session(coordinator.engine) as session:
        failed = session.scalars(
            select(IndexOperation).where(IndexOperation.status == "FAILED")
        ).one()
        assert failed.plan is not None
        assert "changed" not in session.scalars(select(Document)).one().normalized_payload["text"]
    before = len(encoder.calls)
    coordinator.boundary = lambda _: None
    recovered = coordinator.run(second, repository="acme/repo")
    assert recovered.status == "READY" and recovered.attempts == 2
    assert len(coordinator.ready("acme/repo").ids) == 1
    if stage != "prepared":
        assert len(encoder.calls) == before
    assert coordinator.store.client.count(coordinator.store.collection, exact=True).count == 2


def test_partial_real_vector_upsert_recovers(application, tmp_path, monkeypatch):
    coordinator, _, _ = application
    input_path = snapshot(tmp_path, [doc(), doc("2", "other")])
    real_upsert = coordinator.store.upsert

    def partial(chunks, vectors, dimension):
        real_upsert(chunks[:1], vectors[:1], dimension)
        raise RuntimeError("interrupted after first real vector")

    monkeypatch.setattr(coordinator.store, "upsert", partial)
    with pytest.raises(IndexingError):
        coordinator.run(input_path, repository="acme/repo")
    monkeypatch.setattr(coordinator.store, "upsert", real_upsert)
    coordinator.run(input_path, repository="acme/repo")
    assert coordinator.store.client.count(coordinator.store.collection, exact=True).count == 2
    assert len(coordinator.ready("acme/repo").vector_search([1, 0, 0])) == 2


def test_multi_repository_collision_isolation_and_serialization(application, tmp_path):
    coordinator, _, _ = application
    a = coordinator.run(
        snapshot(tmp_path, [doc(), doc(family="pull_request")]), repository="acme/repo"
    )
    b = coordinator.run(
        snapshot(tmp_path, [doc(repository="acme/other")], name="acme/other"),
        repository="acme/other",
    )
    assert len(coordinator.ready("acme/repo").ids) == 2
    assert not set(coordinator.ready("acme/repo").ids) & set(coordinator.ready("acme/other").ids)
    assert a.repository_id != b.repository_id
    lock = int.from_bytes(hashlib.sha256(b"acme/repo").digest()[:8], "big", signed=True)
    with coordinator.engine.connect() as conn:
        conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": lock})
        conn.commit()
        try:
            with pytest.raises(IndexingError, match="lock"):
                coordinator.run(snapshot(tmp_path, [doc()]), repository="acme/repo")
            coordinator.run(
                snapshot(tmp_path, [doc(repository="acme/other")], name="acme/other"),
                repository="acme/other",
            )
        finally:
            conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": lock})
            conn.commit()


def test_checksum_scope_and_missing_report_rejected(application, tmp_path):
    coordinator, _, _ = application
    path = snapshot(tmp_path, [doc()])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        coordinator.run(path, repository="acme/repo")
    path = snapshot(tmp_path, [doc()])
    report = json.loads(report_path(path).read_text())
    report["sources"]["issue"]["observed_source_keys"] = []
    report_path(path).write_text(json.dumps(report))
    with pytest.raises(ValueError, match="identity"):
        coordinator.run(path, repository="acme/repo")
    report_path(path).unlink()
    with pytest.raises(FileNotFoundError):
        coordinator.run(path, repository="acme/repo")


def test_ready_vector_corruption_fails_closed(application, tmp_path):
    coordinator, _, _ = application
    coordinator.run(snapshot(tmp_path, [doc()]), repository="acme/repo")
    ready = coordinator.ready("acme/repo")
    coordinator.store.delete(ready.ids)
    with pytest.raises(IndexingError, match="unavailable"):
        ready.vector_search([1, 0, 0])


def test_identity_includes_typed_repository_source_and_content():
    repository = uuid4()
    base = (repository, "issue", "1", 0, "digest", "model-v1", "chunk-v1")
    assert chunk_identity(*base) == chunk_identity(*base)
    assert chunk_identity(*base) != chunk_identity(repository, "pull_request", *base[2:])
    assert chunk_identity(*base) != chunk_identity(uuid4(), *base[1:])


def test_retirement_deletes_only_stale_points_and_replays_partial_deletion(application, tmp_path):
    coordinator, _, _ = application
    first = coordinator.run(
        snapshot(tmp_path, [doc("1"), doc("2", "other"), doc("3", "third")]), repository="acme/repo"
    )
    coordinator.run(
        snapshot(tmp_path, [doc("1"), doc("2", "changed other"), doc("3", "changed third")]),
        repository="acme/repo",
    )
    current = coordinator.ready("acme/repo")
    assert coordinator.store.client.count(coordinator.store.collection, exact=True).count == 5
    with pytest.raises(IndexingError, match="current"):
        coordinator.retire_revision(current.revision, repository="acme/repo")
    coordinator.store.batch_size = 1

    def fail(_):
        raise RuntimeError("partial deletion")

    coordinator.boundary = fail
    with pytest.raises(RuntimeError):
        coordinator.retire_revision(first.id, repository="acme/repo")
    with Session(coordinator.engine) as session:
        assert session.get(IndexOperation, first.id).status == "RETIRING"
    assert len(current.vector_search([1, 0, 0])) == 3
    coordinator.boundary = lambda _: None
    coordinator.retire_revision(first.id, repository="acme/repo")
    coordinator.retire_revision(first.id, repository="acme/repo")
    assert coordinator.store.client.count(coordinator.store.collection, exact=True).count == 3
    assert set(coordinator.store.lookup(current.ids)) == set(current.ids)


def test_concurrent_metadata_import_detected_and_failed_plan_can_be_retired(application, tmp_path):
    from github_intelligence.persistence.operations import import_documents

    coordinator, _, _ = application
    initial = coordinator.run(snapshot(tmp_path, [doc()]), repository="acme/repo")
    candidate = snapshot(tmp_path, [doc(body="changed candidate")])

    def import_during_prepare(stage):
        if stage == "prepared":
            with Session(coordinator.engine) as session, session.begin():
                import_documents(session, [doc(body="independent metadata import")])

    coordinator.boundary = import_during_prepare
    with pytest.raises(IndexingError):
        coordinator.run(candidate, repository="acme/repo")
    assert coordinator.ready("acme/repo").revision == initial.id
    with Session(coordinator.engine) as session:
        assert (
            session.scalars(select(Document)).one().normalized_payload["text"]
            == "independent metadata import"
        )
        failed = session.scalars(
            select(IndexOperation).where(IndexOperation.status == "FAILED")
        ).one()
        failed_id = failed.id
    coordinator.boundary = lambda _: None
    coordinator.retire_revision(failed_id, repository="acme/repo")
    recovered = coordinator.run(
        snapshot(tmp_path, [doc(body="new observation")]), repository="acme/repo"
    )
    assert recovered.status == "READY"


def test_real_process_interruption_releases_lock_and_replays(application, tmp_path):
    import os
    import subprocess
    import sys

    coordinator, _, _ = application
    original = coordinator.run(snapshot(tmp_path, [doc()]), repository="acme/repo")
    candidate = snapshot(tmp_path, [doc(body="changed after process crash")])
    vector_path = coordinator.store.path
    coordinator.store.client.close()
    program = """
import os, sys
from pathlib import Path
from github_intelligence.persistence.database import create_database_engine
from github_intelligence.indexing.coordinator import IncrementalCoordinator
from github_intelligence.indexing.storage import IncrementalVectorStore
from github_intelligence.retrieval import DocumentChunker
class Encoder:
    dimension = 3
    def encode(self, texts): return [[1.0, 1.0, 0.0] for _ in texts]
def crash(stage):
    if stage == "vectors_written": os._exit(7)
coordinator = IncrementalCoordinator(
    create_database_engine(os.environ["TEST_DATABASE_URL"]),
    IncrementalVectorStore(sys.argv[1], "application"), Encoder(), Path(sys.argv[2]),
    embedding_version="test-v1", chunker=DocumentChunker(20, 0), boundary=crash,
)
coordinator.run(Path(sys.argv[3]), repository="acme/repo")
"""
    env = os.environ.copy()
    env["TEST_DATABASE_URL"] = coordinator.engine.url.render_as_string(hide_password=False)
    result = subprocess.run(
        [sys.executable, "-c", program, vector_path, str(coordinator.root), str(candidate)],
        env=env,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 7, "Child did not reach the crash boundary"
    coordinator.store = IncrementalVectorStore(vector_path, "application")
    with Session(coordinator.engine) as session:
        pending = session.scalars(
            select(IndexOperation).where(IndexOperation.status == "VECTOR_WRITES")
        ).one()
        assert pending.plan is not None
    assert coordinator.ready("acme/repo").revision == original.id
    recovered = coordinator.run(candidate, repository="acme/repo")
    assert recovered.status == "READY" and recovered.attempts == 2
    assert coordinator.store.client.count("application", exact=True).count == 2


def test_same_tree_conflict_cached_and_unknown_scope_preserve(application, tmp_path):
    coordinator, _, _ = application
    coordinator.run(
        snapshot(tmp_path, [doc("README.md", family="documentation")]), repository="acme/repo"
    )
    for mode in ["same_tree", "cached", "filter"]:
        path = snapshot(tmp_path, [], head_sha="d" * 40, tree_sha="e" * 40)
        report = IngestionReport.model_validate_json(report_path(path).read_text())
        if mode == "same_tree":
            report.repository.tree_sha = "c" * 40
            report.sources["documentation"].scope["tree_sha"] = "c" * 40
        elif mode == "cached":
            report.fresh = False
        else:
            report.sources["documentation"].scope["prefix"] = "docs/"
        report_path(path).write_text(report.model_dump_json())
        result = coordinator.run(path, repository="acme/repo", reconcile=True)
        assert result.plan["counts"]["deleted"] == 0 and result.plan["preserved"]


def test_cli_opt_in_real_stores_and_existing_index_protected(
    migrated_engine, tmp_path, monkeypatch, capsys
):
    from github_intelligence import cli
    from github_intelligence.config import Settings

    settings = Settings(
        _env_file=None,
        database_url=migrated_engine.url.render_as_string(hide_password=False),
        index_dir=tmp_path / "benchmark-index",
        qdrant_path=tmp_path / "benchmark-vectors",
    )
    settings.index_dir.mkdir()
    original = settings.index_dir / "chunks.jsonl"
    original.write_text("frozen marker")
    monkeypatch.setattr(cli, "SentenceTransformerEncoder", lambda *_: CountingEncoder())
    parser = cli.build_parser()
    path = snapshot(tmp_path, [doc()])
    root = tmp_path / "application"
    args = parser.parse_args(
        [
            "index-incremental",
            str(path),
            "--repo",
            "acme/repo",
            "--root",
            str(root),
            "--embedding-version",
            "test-v1",
        ]
    )
    assert cli.run_incremental(args, settings) == 0
    output = capsys.readouterr().out
    assert "new=1" in output and "status=READY" in output
    assert cli.run_incremental(args, settings) == 0
    assert original.read_text() == "frozen marker"
    args.root = settings.index_dir
    assert cli.run_incremental(args, settings) == 2
    assert original.read_text() == "frozen marker"
    args.root = root
    args.input = tmp_path / "missing.jsonl"
    assert cli.run_incremental(args, settings) == 1


def test_prior_metadata_import_is_indexed_without_assuming_manifest(application, tmp_path):
    from github_intelligence.persistence.operations import import_documents

    coordinator, encoder, _ = application
    with Session(coordinator.engine) as session, session.begin():
        import_documents(session, [doc()])
    result = coordinator.run(snapshot(tmp_path, [doc()]), repository="acme/repo")
    assert result.plan["counts"]["unchanged"] == 1
    assert encoder.calls and coordinator.ready("acme/repo").ids
    with Session(coordinator.engine) as session:
        manifest = session.scalars(select(ChunkManifest)).one()
        assert manifest.document_identity == session.scalars(select(Document.id)).one()
