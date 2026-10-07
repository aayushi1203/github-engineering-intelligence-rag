import base64
import json
from pathlib import Path

import httpx
import pytest

from github_intelligence.config import Settings
from github_intelligence.github.client import GitHubClient
from github_intelligence.github.scans import ScanStatus
from github_intelligence.ingestion.models import IngestionReport
from github_intelligence.ingestion.service import (
    IngestionError,
    IngestionOptions,
    RepositoryIngestor,
)
from github_intelligence.ingestion.snapshot import (
    load_snapshot_report,
    publish_snapshot,
    report_path,
)
from github_intelligence.models import DocumentType
from github_intelligence.persistence.identity import document_id

HEAD, TREE, BLOB = "a" * 40, "b" * 40, "c" * 40


class GitHubFixture:
    def __init__(self):
        self.requests = []
        self.truncated = False
        self.limit_issues = False
        self.fail_endpoint = None
        self.bad_blob = False
        self.too_large = False
        self.empty_review = False
        self.duplicate_issues = False

    def __call__(self, request):
        self.requests.append(request)
        path = request.url.path
        if path == self.fail_endpoint:
            return httpx.Response(500, json={"message": "dummy-sensitive-error"})
        bodies = {
            "/repos/acme/repo": {
                "id": 42,
                "full_name": "Acme/Repo",
                "html_url": "https://github.com/acme/repo",
                "default_branch": "feature/work",
                "private": False,
            },
            "/repos/acme/repo/commits/feature/work": {
                "sha": HEAD,
                "commit": {"tree": {"sha": TREE}},
            },
            "/repos/acme/repo/issues": [
                {
                    "id": 1,
                    "number": 1,
                    "title": "Bug",
                    "body": "Timeout bug",
                    "html_url": "https://github.com/acme/repo/issues/1",
                    "updated_at": "2026-01-02T00:00:00Z",
                }
            ],
            "/repos/acme/repo/issues/1/comments": [
                {
                    "id": 11,
                    "body": "Issue comment",
                    "html_url": "https://github.com/acme/repo/issues/1#issuecomment-11",
                }
            ],
            "/repos/acme/repo/pulls": [
                {
                    "id": 2,
                    "number": 2,
                    "title": "Fix",
                    "body": "Fix timeout",
                    "html_url": "https://github.com/acme/repo/pull/2",
                    "head": {"ref": "fix", "sha": HEAD},
                }
            ],
            "/repos/acme/repo/issues/2/comments": [
                {
                    "id": 21,
                    "body": "PR conversation",
                    "html_url": "https://github.com/acme/repo/pull/2#issuecomment-21",
                }
            ],
            "/repos/acme/repo/pulls/2/comments": [
                {
                    "id": 22,
                    "body": "Inline code feedback",
                    "html_url": "https://github.com/acme/repo/pull/2#discussion_r22",
                    "pull_request_review_id": 23,
                }
            ],
            "/repos/acme/repo/pulls/2/reviews": [
                {
                    "id": 23,
                    "body": "" if self.empty_review else "Review summary",
                    "state": "APPROVED",
                    "user": {"login": "reviewer"},
                    "commit_id": HEAD,
                    "submitted_at": "2026-01-03T00:00:00Z",
                    "html_url": "https://github.com/acme/repo/pull/2#pullrequestreview-23",
                }
            ],
            "/repos/acme/repo/commits": [
                {
                    "sha": HEAD,
                    "html_url": f"https://github.com/acme/repo/commit/{HEAD}",
                    "commit": {
                        "message": "Fix timeout",
                        "author": {"date": "2026-01-01T00:00:00Z"},
                    },
                }
            ],
            "/repos/acme/repo/releases": [],
            f"/repos/acme/repo/git/trees/{TREE}": {
                "sha": TREE,
                "truncated": self.truncated,
                "tree": [
                    {
                        "type": "blob",
                        "path": "docs/guide.md",
                        "sha": BLOB,
                        "size": 9999 if self.too_large else 30,
                    }
                ],
            },
            f"/repos/acme/repo/git/blobs/{BLOB}": {
                "sha": BLOB,
                "encoding": "base64",
                "content": "invalid!"
                if self.bad_blob
                else base64.b64encode(b"# Guide\n\nUse timeout safely.").decode(),
            },
        }
        assert path in bodies, path
        headers = {}
        if path.endswith("/issues") and self.duplicate_issues:
            bodies[path] += bodies[path]
        if path.endswith("/issues") and self.limit_issues:
            headers["link"] = '<https://api.github.com/repos/acme/repo/issues?page=2>; rel="next"'
        return httpx.Response(200, json=bodies[path], headers=headers)


def settings(tmp_path: Path, **updates) -> Settings:
    return Settings(
        _env_file=None,
        github_token=None,
        github_cache_dir=tmp_path / "cache",
        github_max_retries=1,
        **updates,
    )


@pytest.mark.asyncio
async def test_complete_families_checkpoint_provenance_and_immutable_docs(tmp_path: Path) -> None:
    fixture = GitHubFixture()
    async with GitHubClient(settings(tmp_path), httpx.MockTransport(fixture)) as client:
        ingestor = RepositoryIngestor(client)
        docs = [doc async for doc in ingestor.ingest("Acme/Repo")]
        report = ingestor.report
    assert report.full_snapshot and report.complete_for_requested_scope and report.fresh
    assert all(scan.status == ScanStatus.COMPLETE for scan in report.sources.values())
    assert (
        report.repository.github_id == "42" and report.repository.default_branch == "feature/work"
    )
    assert report.repository.head_sha == HEAD and report.repository.tree_sha == TREE
    cp = report.sources["issue"].checkpoints[0]
    assert cp.last_source_id == "1" and cp.next_page is None and cp.pages_fetched == 1
    assert cp.max_source_updated_at.isoformat() == "2026-01-02T00:00:00+00:00"
    doc = next(d for d in docs if d.document_type == DocumentType.DOCUMENTATION)
    assert doc.github_id == BLOB and doc.metadata["branch"] == "feature/work"
    assert doc.metadata["commit_sha"] == HEAD
    assert f"/blob/{HEAD}/docs/guide.md" in str(doc.source_url)
    assert doc.metadata["source_key"] == '["feature/work","docs/guide.md"]'
    assert all(d.metadata["repository_github_id"] == "42" for d in docs)
    assert not any("/contents/" in r.url.path for r in fixture.requests)
    assert (
        next(r for r in fixture.requests if r.url.path.endswith("/commits")).url.params["sha"]
        == HEAD
    )
    assert IngestionReport.model_validate_json(report.model_dump_json()) == report


@pytest.mark.asyncio
async def test_reviews_preserve_summary_without_repeating_inline_content(tmp_path: Path) -> None:
    fixture = GitHubFixture()
    async with GitHubClient(settings(tmp_path), httpx.MockTransport(fixture)) as client:
        docs = [doc async for doc in RepositoryIngestor(client).ingest("acme/repo")]
    review = next(doc for doc in docs if doc.document_type == DocumentType.PULL_REQUEST_REVIEW)
    inline = next(
        doc for doc in docs if doc.document_type == DocumentType.PULL_REQUEST_REVIEW_COMMENT
    )
    assert review.text == "Review summary" and inline.text == "Inline code feedback"
    assert review.metadata["author"] == "reviewer" and review.metadata["parent_number"] == 2
    assert inline.metadata["pull_request_review_id"] == int(review.github_id)
    assert review.created_at.isoformat() == "2026-01-03T00:00:00+00:00"
    assert document_id(review) != document_id(inline)


@pytest.mark.asyncio
async def test_empty_review_membership_remains_observed(tmp_path: Path) -> None:
    fixture = GitHubFixture()
    fixture.empty_review = True
    async with GitHubClient(settings(tmp_path), httpx.MockTransport(fixture)) as client:
        ingestor = RepositoryIngestor(client)
        docs = [doc async for doc in ingestor.ingest("acme/repo")]
    scan = ingestor.report.sources["pull_request_review"]
    assert scan.status == ScanStatus.COMPLETE and scan.observed_source_keys == ["23"]
    assert scan.skipped_empty == 1 and not any(
        d.document_type == DocumentType.PULL_REQUEST_REVIEW for d in docs
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flag,status",
    [
        ("truncated", ScanStatus.TRUNCATED),
        ("too_large", ScanStatus.LIMITED),
        ("duplicate_issues", ScanStatus.LIMITED),
    ],
)
async def test_incomplete_scans_are_explicit_and_not_published_by_default(
    tmp_path: Path, flag: str, status: ScanStatus
) -> None:
    fixture = GitHubFixture()
    setattr(fixture, flag, True)
    async with GitHubClient(settings(tmp_path), httpx.MockTransport(fixture)) as client:
        ingestor = RepositoryIngestor(client, doc_max_bytes=100)
        output = tmp_path / "snapshot.jsonl"
        with pytest.raises(IngestionError, match="incomplete_requested_scope"):
            await publish_snapshot(ingestor, "acme/repo", output)
        assert not output.exists() and not report_path(output).exists()
        family = "issue" if flag == "duplicate_issues" else "documentation"
        assert ingestor.report.sources[family].status == status
        report = await publish_snapshot(ingestor, "acme/repo", output, allow_incomplete=True)
        assert not report.full_snapshot and not report.complete_for_requested_scope
        assert load_snapshot_report(output) == report
        assert not list(tmp_path.glob(".ingestion-*"))


@pytest.mark.asyncio
async def test_page_limit_propagates_to_child_comment_scope(tmp_path: Path) -> None:
    fixture = GitHubFixture()
    fixture.limit_issues = True
    async with GitHubClient(
        settings(tmp_path, github_max_pages=1), httpx.MockTransport(fixture)
    ) as client:
        ingestor = RepositoryIngestor(client)
        _ = [doc async for doc in ingestor.ingest("acme/repo")]
    assert ingestor.report.sources["issue"].status == ScanStatus.LIMITED
    assert ingestor.report.sources["issue_comment"].status == ScanStatus.LIMITED
    assert ingestor.report.sources["issue"].checkpoints[0].next_page == 2
    assert not ingestor.report.complete_for_requested_scope


@pytest.mark.asyncio
async def test_excluded_sources_are_scoped_not_full_and_cache_choice_recorded(
    tmp_path: Path,
) -> None:
    fixture = GitHubFixture()
    async with GitHubClient(settings(tmp_path), httpx.MockTransport(fixture)) as client:
        report = await publish_snapshot(
            RepositoryIngestor(client),
            "acme/repo",
            tmp_path / "snapshot.jsonl",
            IngestionOptions(
                include_comments=False,
                include_documentation=False,
                include_reviews=False,
                fresh=False,
            ),
        )
    assert report.complete_for_requested_scope and not report.full_snapshot and not report.fresh
    assert report.sources["documentation"].reason == "excluded_by_options"
    assert not any(
        "/comments" in r.url.path or "/git/" in r.url.path or "/reviews" in r.url.path
        for r in fixture.requests
    )


@pytest.mark.asyncio
async def test_failed_ingestion_keeps_previous_snapshot_and_report_unchanged(
    tmp_path: Path,
) -> None:
    fixture = GitHubFixture()
    output = tmp_path / "snapshot.jsonl"
    async with GitHubClient(settings(tmp_path), httpx.MockTransport(fixture)) as client:
        ingestor = RepositoryIngestor(client)
        await publish_snapshot(ingestor, "acme/repo", output)
        previous = output.read_bytes(), report_path(output).read_bytes()
        fixture.fail_endpoint = "/repos/acme/repo/pulls/2/comments"
        with pytest.raises(IngestionError) as caught:
            await publish_snapshot(ingestor, "acme/repo", output, allow_incomplete=True)
        assert (
            caught.value.report.sources["pull_request_review_comment"].status == ScanStatus.FAILED
        )
        assert "dummy-sensitive" not in str(caught.value)
    assert (output.read_bytes(), report_path(output).read_bytes()) == previous
    assert not list(tmp_path.glob(".ingestion-*"))


@pytest.mark.asyncio
async def test_bad_blob_and_revision_mismatch_are_failures_not_empty_docs(tmp_path: Path) -> None:
    fixture = GitHubFixture()
    fixture.bad_blob = True
    async with GitHubClient(settings(tmp_path), httpx.MockTransport(fixture)) as client:
        ingestor = RepositoryIngestor(client)
        with pytest.raises(IngestionError, match="invalid_documentation_encoding"):
            await publish_snapshot(ingestor, "acme/repo", tmp_path / "snapshot.jsonl")
        assert ingestor.report.sources["documentation"].status == ScanStatus.FAILED
    assert not (tmp_path / "snapshot.jsonl").exists()


@pytest.mark.asyncio
async def test_successful_snapshot_report_digest_and_interrupted_pair_detection(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = GitHubFixture()
    output = tmp_path / "snapshot.jsonl"
    async with GitHubClient(settings(tmp_path), httpx.MockTransport(fixture)) as client:
        report = await publish_snapshot(RepositoryIngestor(client), "acme/repo", output)
    assert load_snapshot_report(output) == report
    output.write_text('{"different":"snapshot"}\n')
    with pytest.raises(ValueError, match="mismatched"):
        load_snapshot_report(output)
    assert all("dummy" not in line for line in report_path(output).read_text().splitlines())
    assert (
        len([json.loads(line) for line in (tmp_path / "snapshot.jsonl").read_text().splitlines()])
        == 1
    )


@pytest.mark.asyncio
async def test_abandoned_ingestion_cannot_claim_completeness(tmp_path: Path) -> None:
    async with GitHubClient(settings(tmp_path), httpx.MockTransport(GitHubFixture())) as client:
        ingestor = RepositoryIngestor(client)
        iterator = ingestor.ingest("acme/repo")
        await anext(iterator)
        await iterator.aclose()
        assert (
            not ingestor.report.full_snapshot and ingestor.report.error_code == "scan_interrupted"
        )


@pytest.mark.asyncio
async def test_blob_sha_mismatch_and_tree_shape_failure(tmp_path: Path) -> None:
    fixture = GitHubFixture()

    def handler(request):
        response = fixture(request)
        if request.url.path.endswith(f"/blobs/{BLOB}"):
            body = response.json()
            body["sha"] = "d" * 40
            return httpx.Response(200, json=body)
        return response

    async with GitHubClient(settings(tmp_path), httpx.MockTransport(handler)) as client:
        ingestor = RepositoryIngestor(client)
        with pytest.raises(IngestionError, match="blob_revision_mismatch"):
            await publish_snapshot(ingestor, "acme/repo", tmp_path / "output.jsonl")
    assert not (tmp_path / "output.jsonl").exists()
    failed = IngestionReport.model_validate_json(
        (tmp_path / "output.jsonl.failed-scan.json").read_text()
    )
    assert failed.sources["documentation"].status == ScanStatus.FAILED
    assert failed.snapshot_sha256 is None and not failed.full_snapshot


@pytest.mark.asyncio
async def test_interruption_between_report_and_snapshot_is_detected(
    tmp_path: Path, monkeypatch
) -> None:
    import os

    fixture = GitHubFixture()
    output = tmp_path / "output.jsonl"
    output.write_text("previous snapshot")
    real_replace = os.replace

    def interrupted(source, destination):
        if Path(destination) == output:
            raise OSError("publication interrupted")
        real_replace(source, destination)

    monkeypatch.setattr("github_intelligence.ingestion.snapshot.os.replace", interrupted)
    async with GitHubClient(settings(tmp_path), httpx.MockTransport(fixture)) as client:
        with pytest.raises(OSError):
            await publish_snapshot(RepositoryIngestor(client), "acme/repo", output)
    assert output.read_text() == "previous snapshot"
    with pytest.raises(ValueError, match="mismatched"):
        load_snapshot_report(output)
    assert not list(tmp_path.glob(".ingestion-*"))


@pytest.mark.asyncio
async def test_cli_publication_and_failure_exit_codes(tmp_path: Path, monkeypatch, capsys) -> None:
    from github_intelligence.cli import build_parser, run_ingest

    fixture = GitHubFixture()

    def create_client(config):
        return GitHubClient(config, httpx.MockTransport(fixture))

    monkeypatch.setattr("github_intelligence.cli.GitHubClient", create_client)
    output = tmp_path / "output.jsonl"
    args = build_parser().parse_args(["ingest", "acme/repo", "--output", str(output)])
    assert await run_ingest(args, settings(tmp_path)) == 0
    assert "(full)" in capsys.readouterr().out
    fixture.fail_endpoint = "/repos/acme/repo/releases"
    assert await run_ingest(args, settings(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "No successful snapshot" in captured.err and "dummy-sensitive" not in captured.err
    assert load_snapshot_report(output).full_snapshot
