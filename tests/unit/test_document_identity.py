import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from github_intelligence.cli import build_parser, run_import_metadata
from github_intelligence.config import Settings
from github_intelligence.models import GitHubDocument
from github_intelligence.persistence.identity import (
    canonical_repository,
    content_hash,
    document_id,
    normalized_payload,
    source_key,
)


def document(**updates: object) -> GitHubDocument:
    data: dict[str, object] = {
        "repository": "Acme/Repo",
        "document_type": "issue",
        "github_id": "123",
        "title": "Fix timeout",
        "text": "Fixed timeout behavior",
        "source_url": "https://github.com/acme/repo/issues/1",
        "updated_at": "2026-01-01T00:00:00Z",
        "metadata": {"author": "dev"},
    }
    data.update(updates)
    return GitHubDocument.model_validate(data)


def test_identity_is_deterministic_case_normalized_and_typed() -> None:
    original = document()
    assert document_id(original) == document_id(document(repository="acme/repo"))
    assert document_id(original) != document_id(document(document_type="issue_comment"))
    assert document_id(original) != document_id(document(repository="acme/other"))
    assert document_id(original) == document_id(document(text="edited"))


def test_documentation_identity_survives_sha_changes_and_distinguishes_paths() -> None:
    first = document(
        document_type="documentation",
        github_id="blob1",
        source_url="https://github.com/acme/repo/blob/feature/ref/docs/a.md",
        metadata={"path": "docs/a.md"},
    )
    edited = first.model_copy(update={"github_id": "blob2", "text": "edited"})
    other = document(
        document_type="documentation",
        github_id="blob1",
        source_url="https://github.com/acme/repo/blob/main/docs/b.md",
        metadata={"path": "docs/b.md", "branch": "main"},
    )
    assert source_key(first) == '["feature/ref","docs/a.md"]'
    assert document_id(first) == document_id(edited)
    assert content_hash(first) != content_hash(edited)
    assert document_id(first) != document_id(other)
    assert document_id(first) != document_id(
        first.model_copy(update={"metadata": {"path": "docs/a.md", "branch": "main"}})
    )


def test_hash_ignores_operational_metadata_and_key_order_but_detects_source_edits() -> None:
    first = document(metadata={"author": "dev", "labels": ["bug"]})
    same = document(
        metadata={
            "index_status": "indexed",
            "labels": ["bug"],
            "author": "dev",
            "last_seen_at": "tomorrow",
        }
    )
    assert content_hash(first) == content_hash(same)
    assert normalized_payload(same)["metadata"] == {"author": "dev", "labels": ["bug"]}
    assert content_hash(first) != content_hash(document(title="new title"))
    assert content_hash(first) != content_hash(document(metadata={"author": "other"}))
    assert content_hash(document()) == content_hash(
        document(updated_at="2025-12-31T19:00:00-05:00")
    )


@pytest.mark.parametrize(
    "metadata",
    [{"GitHub-Token": "dummy"}, {"nested": [{"api_key": "dummy"}]}, {"client_secret": "dummy"}],
)
def test_credentials_rejected_without_echoing_values(metadata: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Credential-bearing") as error:
        content_hash(document(metadata=metadata))
    assert "dummy" not in str(error.value)


def test_invalid_documentation_identity_and_naive_times_fail() -> None:
    with pytest.raises(ValueError, match="relative"):
        document_id(document(document_type="documentation", metadata={"path": "../a.md"}))
    with pytest.raises(ValueError, match="timezone"):
        content_hash(document(updated_at=datetime(2026, 1, 1)))
    with pytest.raises(ValueError):
        canonical_repository("owner/../repo")


def test_metadata_cli_is_explicit_and_missing_database_is_sanitized(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = build_parser().parse_args(["import-metadata", "data/corpus/benchmark_v2.jsonl"])
    assert run_import_metadata(args, Settings(_env_file=None, database_url=None)) == 1
    assert "DATABASE_URL is required" in capsys.readouterr().err
    assert "dummy-credential" not in repr(Settings(_env_file=None, database_url="dummy-credential"))


def test_existing_cli_and_bm25_work_without_optional_dependencies(tmp_path: Path) -> None:
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sqlalchemy','alembic','psycopg'}:
            raise ImportError('optional persistence is unavailable')
sys.meta_path.insert(0, Block())
from github_intelligence.cli import build_parser
from github_intelligence.config import Settings
from github_intelligence.retrieval import BM25Retriever
assert 'import-metadata' in build_parser().format_help()
assert BM25Retriever(__import__('pathlib').Path(sys.argv[1])).search('timeout') == []
assert Settings(_env_file=None, database_url=None).database_url is None
from github_intelligence.mcp.server import create_mcp_server
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "chunks.jsonl")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
