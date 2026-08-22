from github_intelligence.models import DocumentType, GitHubDocument
from github_intelligence.retrieval.chunking import DocumentChunker


def document(text: str, kind: DocumentType = DocumentType.DOCUMENTATION) -> GitHubDocument:
    return GitHubDocument(
        text=text,
        document_type=kind,
        repository="encode/httpx",
        title="Authentication guide",
        source_url="https://github.com/encode/httpx/blob/main/docs/auth.md",
        github_id="abc123",
        metadata={"parent_number": 42},
    )


def test_documentation_chunks_on_headings_and_preserves_context() -> None:
    chunks = DocumentChunker(max_words=30, overlap_words=5).chunk(
        document("# Authentication\n\nUse a token.\n\n## Timeouts\n\nRefresh before expiry.")
    )
    assert len(chunks) == 2
    assert "Authentication guide" in chunks[0].text
    assert "Timeouts" in chunks[1].text
    assert chunks[1].metadata["section_heading"] == "Timeouts"
    assert str(chunks[0].source_url).endswith("docs/auth.md")


def test_comment_keeps_parent_metadata_and_title() -> None:
    chunk = DocumentChunker(max_words=30, overlap_words=5).chunk(
        document("The timeout began after v1.2.", DocumentType.ISSUE_COMMENT)
    )[0]
    assert chunk.metadata["parent_number"] == 42
    assert chunk.title == "Authentication guide"
    assert chunk.source_type == DocumentType.ISSUE_COMMENT


def test_issue_title_is_context_not_duplicated() -> None:
    chunk = DocumentChunker(max_words=30, overlap_words=5).chunk(
        document(
            "Authentication guide\n\nThe token refresh times out.", DocumentType.ISSUE
        )
    )[0]
    assert chunk.text.count("Authentication guide") == 1
