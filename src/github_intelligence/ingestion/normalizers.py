from typing import Any, cast

from pydantic import HttpUrl

from github_intelligence.models import DocumentType, GitHubDocument


def _user(payload: dict[str, Any]) -> str | None:
    user = payload.get("user") or payload.get("author")
    return user.get("login") if isinstance(user, dict) else None


def _text(*parts: Any) -> str:
    return (
        "\n\n".join(str(part).strip() for part in parts if part and str(part).strip())
        or "(no body)"
    )


def normalize_issue(repo: str, item: dict[str, Any]) -> GitHubDocument:
    labels = [label["name"] for label in item.get("labels", []) if isinstance(label, dict)]
    return GitHubDocument(
        text=_text(item.get("title"), item.get("body")),
        document_type=DocumentType.ISSUE,
        repository=repo,
        title=item["title"],
        source_url=item["html_url"],
        github_id=str(item["id"]),
        created_at=item.get("created_at"),
        updated_at=item.get("updated_at"),
        metadata={
            "number": item["number"],
            "state": item.get("state"),
            "author": _user(item),
            "labels": labels,
            "comments": item.get("comments", 0),
        },
    )


def normalize_comment(
    repo: str, item: dict[str, Any], parent_number: int, *, pull_request: bool, review: bool = False
) -> GitHubDocument:
    kind = (
        DocumentType.PULL_REQUEST_REVIEW_COMMENT
        if review
        else DocumentType.PULL_REQUEST_COMMENT
        if pull_request
        else DocumentType.ISSUE_COMMENT
    )
    parent = "PR" if pull_request else "Issue"
    return GitHubDocument(
        text=_text(item.get("body")),
        document_type=kind,
        repository=repo,
        title=f"{parent} #{parent_number} comment by {_user(item) or 'unknown'}",
        source_url=item["html_url"],
        github_id=str(item["id"]),
        created_at=item.get("created_at"),
        updated_at=item.get("updated_at"),
        metadata={
            "parent_number": parent_number,
            "author": _user(item),
            "review_comment": review,
            "path": item.get("path"),
            "commit_id": item.get("commit_id"),
        },
    )


def normalize_pull_request(repo: str, item: dict[str, Any]) -> GitHubDocument:
    return GitHubDocument(
        text=_text(item.get("title"), item.get("body")),
        document_type=DocumentType.PULL_REQUEST,
        repository=repo,
        title=item["title"],
        source_url=item["html_url"],
        github_id=str(item["id"]),
        created_at=item.get("created_at"),
        updated_at=item.get("updated_at"),
        metadata={
            "number": item["number"],
            "state": item.get("state"),
            "author": _user(item),
            "merged": item.get("merged_at") is not None,
            "merged_at": item.get("merged_at"),
            "base_ref": (item.get("base") or {}).get("ref"),
            "head_ref": (item.get("head") or {}).get("ref"),
        },
    )


def normalize_commit(repo: str, item: dict[str, Any]) -> GitHubDocument:
    commit = item.get("commit") or {}
    message = commit.get("message") or item["sha"]
    author = commit.get("author") or {}
    return GitHubDocument(
        text=message,
        document_type=DocumentType.COMMIT,
        repository=repo,
        title=message.splitlines()[0],
        source_url=item["html_url"],
        github_id=item["sha"],
        created_at=author.get("date"),
        updated_at=author.get("date"),
        metadata={
            "sha": item["sha"],
            "author": _user(item) or author.get("name"),
            "parents": [p.get("sha") for p in item.get("parents", [])],
        },
    )


def normalize_release(repo: str, item: dict[str, Any]) -> GitHubDocument:
    title = item.get("name") or item.get("tag_name") or f"Release {item['id']}"
    return GitHubDocument(
        text=_text(title, item.get("body")),
        document_type=DocumentType.RELEASE,
        repository=repo,
        title=title,
        source_url=item["html_url"],
        github_id=str(item["id"]),
        created_at=item.get("created_at"),
        updated_at=item.get("published_at") or item.get("created_at"),
        metadata={
            "tag_name": item.get("tag_name"),
            "author": _user(item),
            "draft": item.get("draft", False),
            "prerelease": item.get("prerelease", False),
        },
    )


def normalize_documentation(
    repo: str, path: str, sha: str, html_url: str, content: str
) -> GitHubDocument:
    return GitHubDocument(
        text=content,
        document_type=DocumentType.DOCUMENTATION,
        repository=repo,
        title=path,
        source_url=cast(HttpUrl, html_url),
        github_id=sha,
        metadata={"path": path, "sha": sha},
    )
