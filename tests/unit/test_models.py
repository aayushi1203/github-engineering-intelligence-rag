from github_intelligence.models import DocumentType, GitHubDocument


def test_document_strips_text_and_serializes_url() -> None:
    document = GitHubDocument(
        text="  fixed authentication  ", document_type=DocumentType.ISSUE,
        repository="owner/repo", title="  Auth regression  ",
        source_url="https://github.com/owner/repo/issues/1", github_id="42",
    )
    assert document.text == "fixed authentication"
    assert document.title == "Auth regression"
    assert str(document.source_url).endswith("/issues/1")

