from github_intelligence.ingestion.normalizers import normalize_issue
from github_intelligence.models import DocumentType


def test_normalize_issue_preserves_retrieval_metadata() -> None:
    payload = {
        "id": 101, "number": 7, "title": "Authentication fails", "body": "401 after upgrade",
        "html_url": "https://github.com/acme/api/issues/7", "state": "open",
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-02T00:00:00Z",
        "comments": 2, "user": {"login": "octocat"}, "labels": [{"name": "bug"}],
    }
    result = normalize_issue("acme/api", payload)
    assert result.document_type == DocumentType.ISSUE
    assert "401 after upgrade" in result.text
    assert result.metadata["labels"] == ["bug"]
    assert result.metadata["number"] == 7

