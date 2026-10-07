from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("psycopg")

from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from github_intelligence.cli import build_parser, load_documents, run_import_metadata
from github_intelligence.config import Settings
from github_intelligence.models import GitHubDocument
from github_intelligence.persistence.database import (
    PersistenceError,
    create_database_engine,
    session_factory,
)
from github_intelligence.persistence.models import (
    Answer,
    Document,
    Feedback,
    IngestionJob,
    Repository,
)
from github_intelligence.persistence.operations import (
    JobStatus,
    SourceReference,
    create_answer,
    create_feedback,
    create_job,
    create_query,
    create_user,
    import_documents,
    register_repository,
    transition_job,
)


def document(**updates: object) -> GitHubDocument:
    data = dict(
        repository="acme/repo",
        document_type="issue",
        github_id="123",
        title="Bug",
        text="Timeout bug",
        source_url="https://github.com/acme/repo/issues/1",
        metadata={"author": "dev"},
    )
    data.update(updates)
    return GitHubDocument.model_validate(data)


def test_duplicate_import_is_a_true_noop_and_changed_document_updates_one_row(
    migrated_engine,
) -> None:
    sessions = session_factory(migrated_engine)
    with sessions.begin() as session:
        assert import_documents(session, [document()]).new == 1
    with sessions.begin() as session:
        before = session.scalars(select(Document)).one()
        snapshot = (before.id, before.updated_at, before.last_seen_at, before.normalized_payload)
        before.index_status, before.embedding_version = "indexed", "model-v1"
    with sessions.begin() as session:
        assert import_documents(session, [document()]).unchanged == 1
        row = session.scalars(select(Document)).one()
        assert (row.id, row.updated_at, row.last_seen_at, row.normalized_payload) == snapshot
        assert (row.index_status, row.embedding_version) == ("indexed", "model-v1")
    with sessions.begin() as session:
        assert import_documents(session, [document(text="Fixed timeout bug")]).changed == 1
        row = session.scalars(select(Document)).one()
        assert row.id == snapshot[0]
        assert row.normalized_payload["text"] == "Fixed timeout bug"
        assert row.index_status == "pending" and row.embedding_version is None
        assert session.scalar(select(func.count()).select_from(Document)) == 1


def test_full_frozen_corpus_import_counts_and_round_trip(migrated_engine) -> None:
    corpus = Path(__file__).resolve().parents[2] / "data/corpus/benchmark_v2.jsonl"
    docs = load_documents(corpus)
    with session_factory(migrated_engine).begin() as session:
        assert import_documents(session, docs).new == 892
    with session_factory(migrated_engine).begin() as session:
        result = import_documents(session, docs)
        assert (result.new, result.changed, result.unchanged) == (0, 0, 892)
        assert session.scalar(select(func.count()).select_from(Repository)) == 3
        row = session.scalars(select(Document)).first()
        restored = GitHubDocument.model_validate(row.normalized_payload)
        assert restored.github_id == row.github_id


def test_typed_and_documentation_keys_prevent_collisions(migrated_engine) -> None:
    first = document(
        document_type="documentation",
        github_id="blob1",
        source_url="https://github.com/acme/repo/blob/main/docs/a.md",
        metadata={"path": "docs/a.md", "sha": "blob1"},
    )
    second = document(
        document_type="documentation",
        github_id="blob1",
        source_url="https://github.com/acme/repo/blob/main/docs/b.md",
        metadata={"path": "docs/b.md", "sha": "blob1"},
    )
    with session_factory(migrated_engine).begin() as session:
        assert (
            import_documents(
                session, [document(), document(document_type="issue_comment"), first, second]
            ).new
            == 4
        )
        assert (
            import_documents(
                session,
                [
                    first.model_copy(
                        update={
                            "github_id": "blob2",
                            "text": "edited",
                            "metadata": {"path": "docs/a.md", "sha": "blob2"},
                        }
                    )
                ],
            ).changed
            == 1
        )
        assert session.scalar(select(func.count()).select_from(Document)) == 4


def test_import_rollback_and_validation_before_writes(migrated_engine) -> None:
    sessions = session_factory(migrated_engine)
    with pytest.raises(RuntimeError), sessions.begin() as session:
        import_documents(session, [document()])
        raise RuntimeError("abort")
    with sessions.begin() as session:
        assert session.scalar(select(func.count()).select_from(Document)) == 0
        assert session.scalar(select(func.count()).select_from(Repository)) == 0
        with pytest.raises(ValueError, match="Credential"):
            import_documents(session, [document(), document(metadata={"github_token": "dummy"})])
        assert session.scalar(select(func.count()).select_from(Repository)) == 0


def test_concurrent_duplicate_imports_are_serialized(migrated_engine) -> None:
    def import_once(_: int):
        with session_factory(migrated_engine).begin() as session:
            return import_documents(session, [document()])

    with ThreadPoolExecutor(max_workers=2) as executor:
        counts = list(executor.map(import_once, range(2)))
    assert sum(count.new for count in counts) == 1
    assert sum(count.unchanged for count in counts) == 1


def test_job_idempotency_and_all_lifecycle_transitions(migrated_engine) -> None:
    with session_factory(migrated_engine).begin() as session:
        repo = register_repository(session, "Acme/Repo")
        job = create_job(session, repo.id, idempotency_key="request-1", scope={"source": "issues"})
        duplicate = create_job(
            session, repo.id, idempotency_key="request-1", scope={"source": "issues"}
        )
        assert job.id == duplicate.id
        with pytest.raises(ValueError, match="different"):
            create_job(session, repo.id, idempotency_key="request-1", scope={"source": "commits"})
        with pytest.raises(ValueError, match="transition"):
            transition_job(session, job.id, JobStatus.INDEXING)
        for target in [
            JobStatus.FETCHING,
            JobStatus.PARSING,
            JobStatus.EMBEDDING,
            JobStatus.INDEXING,
            JobStatus.COMPLETE,
        ]:
            transition_job(session, job.id, target, processed_count=1, total_count=1)
        assert job.attempts == 1 and job.started_at and job.completed_at
        assert job.status == "COMPLETE"
        with pytest.raises(ValueError, match="Terminal"):
            transition_job(session, job.id, JobStatus.FETCHING)
    with session_factory(migrated_engine).begin() as session:
        assert session.scalars(select(IngestionJob)).one().status == "COMPLETE"


def test_failed_jobs_and_progress_validation(migrated_engine) -> None:
    with session_factory(migrated_engine).begin() as session:
        repo = register_repository(session, "acme/repo")
        job = create_job(session, repo.id)
        transition_job(session, job.id, JobStatus.FETCHING, processed_count=2, total_count=3)
        with pytest.raises(ValueError, match="progress"):
            transition_job(session, job.id, JobStatus.PARSING, processed_count=1)
        with pytest.raises(ValueError, match="sanitized"):
            transition_job(session, job.id, JobStatus.FAILED, error_code="password=dummy")
        transition_job(session, job.id, JobStatus.FAILED, error_code="source_unavailable")
        assert job.error_code == "source_unavailable" and job.completed_at
        assert "dummy" not in job.error_detail


def test_user_query_answer_feedback_and_default_content_minimization(migrated_engine) -> None:
    with session_factory(migrated_engine).begin() as session:
        user = create_user(session, external_identity="local:test")
        repo = register_repository(session, "acme/repo", owner_user_id=user.id, visibility="public")
        query = create_query(
            session,
            repo.id,
            user_id=user.id,
            request_key="request",
            query_text="sensitive",
            measurements={"tokens": None},
        )
        ref = SourceReference(source_id="S1", github_url="https://github.com/acme/repo/issues/1")
        answer = create_answer(
            session,
            query.id,
            answer_text="sensitive",
            citations=[ref],
            sources=[ref],
            citation_validation="valid",
            limitations=["sensitive"],
        )
        feedback = create_feedback(
            session, answer.id, rating="negative", category="bad_sources", user_id=user.id
        )
        assert query.query_text is None and answer.answer_text is None and answer.limitations == []
        assert answer.source_references[0]["source_id"] == "S1"
        assert feedback.answer_id == answer.id
        retained = create_answer(
            session, query.id, attempt=2, answer_text="opt-in", store_content=True
        )
        assert retained.answer_text == "opt-in"
    with session_factory(migrated_engine).begin() as session:
        assert session.scalar(select(func.count()).select_from(Feedback)) == 1
        assert session.scalar(select(func.count()).select_from(Answer)) == 2


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO feedback (answer_id, rating) VALUES (gen_random_uuid(), 'positive')",
        "INSERT INTO repositories (name,github_url,visibility) VALUES ('acme/repo','https://github.com/acme/repo','invalid')",
        "INSERT INTO repositories (name,github_url) VALUES ('Acme/Repo','https://github.com/acme/repo')",
    ],
)
def test_database_foreign_keys_and_check_constraints(migrated_engine, statement: str) -> None:
    with pytest.raises(IntegrityError), migrated_engine.begin() as conn:
        conn.execute(text(statement))


@pytest.mark.parametrize(
    "mutation",
    ["rating", "category", "attempt", "query_key", "job_status", "job_progress", "document_key"],
)
def test_database_checks_and_uniqueness_bypass_service(migrated_engine, mutation: str) -> None:
    sessions = session_factory(migrated_engine)
    with sessions.begin() as session:
        repo = register_repository(session, "acme/repo")
        import_documents(session, [document()])
        job = create_job(session, repo.id)
        query = create_query(session, repo.id, request_key="key")
        answer = create_answer(session, query.id)
        feedback = create_feedback(session, answer.id, rating="positive")
        ids = (repo.id, job.id, query.id, answer.id, feedback.id)
    with pytest.raises(IntegrityError), sessions.begin() as session:
        if mutation == "rating":
            session.get(Feedback, ids[4]).rating = "invalid"
        elif mutation == "category":
            session.get(Feedback, ids[4]).category = "invalid"
        elif mutation == "attempt":
            create_answer(session, ids[2], attempt=1)
        elif mutation == "query_key":
            create_query(session, ids[0], request_key="key")
        elif mutation == "job_status":
            session.get(IngestionJob, ids[1]).status = "invalid"
        elif mutation == "job_progress":
            session.get(IngestionJob, ids[1]).processed_count = -1
        else:
            row = session.scalars(select(Document)).one()
            session.add(
                Document(
                    id=uuid4(),
                    repository_id=row.repository_id,
                    source_type=row.source_type,
                    source_key=row.source_key,
                    github_id=row.github_id,
                    github_url=row.github_url,
                    title=row.title,
                    content_hash=row.content_hash,
                    normalized_payload=row.normalized_payload,
                )
            )
        session.flush()


def test_cli_import_failure_redaction_and_atomicity(
    migrated_engine, tmp_path: Path, capsys
) -> None:
    settings = Settings(
        _env_file=None, database_url=migrated_engine.url.render_as_string(hide_password=False)
    )
    path = tmp_path / "input.jsonl"
    path.write_text(document().model_dump_json() + "\n")
    args = build_parser().parse_args(["import-metadata", str(path)])
    assert run_import_metadata(args, settings) == 0
    assert "new=1" in capsys.readouterr().out
    assert run_import_metadata(args, settings) == 0
    assert "unchanged=1" in capsys.readouterr().out
    path.write_text(document(text="edited").model_dump_json() + "\n")
    assert run_import_metadata(args, settings) == 0
    assert "changed=1" in capsys.readouterr().out
    path.write_text(document(github_id="456").model_dump_json() + '\n{"invalid":"sensitive"}\n')
    assert run_import_metadata(args, settings) == 1
    assert "sensitive" not in capsys.readouterr().err
    with session_factory(migrated_engine).begin() as session:
        assert session.scalar(select(func.count()).select_from(Document)) == 1
    bad = Settings(_env_file=None, database_url="postgresql://user:dummy@127.0.0.1:1/test")
    path.write_text(document().model_dump_json())
    assert run_import_metadata(args, bad) == 1
    error = capsys.readouterr().err
    assert "dummy" not in error and "No changes were committed" in error


def test_non_postgresql_urls_rejected() -> None:
    with pytest.raises(PersistenceError):
        create_database_engine("sqlite://")


def test_review_source_migration_and_commit_revision_metadata(migrated_engine) -> None:
    review = document(
        document_type="pull_request_review",
        github_id="review-23",
        metadata={"author": "reviewer", "commit_sha": "a" * 40},
    )
    with session_factory(migrated_engine).begin() as session:
        assert import_documents(session, [review]).new == 1
        row = session.scalars(select(Document)).one()
        assert row.source_type == "pull_request_review" and row.commit_sha == "a" * 40
        assert import_documents(session, [review]).unchanged == 1
