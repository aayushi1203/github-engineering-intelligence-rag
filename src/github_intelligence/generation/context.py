import json

from github_intelligence.generation.models import AnswerSource
from github_intelligence.models import RetrievalResult


def build_context(
    results: list[RetrievalResult], *, max_chars: int = 30_000
) -> tuple[str, list[AnswerSource]]:
    """Serialize retrieved evidence as visibly untrusted, source-addressable data."""
    sources: list[AnswerSource] = []
    remaining = max_chars
    for index, result in enumerate(results, start=1):
        if remaining <= 0:
            break
        text = result.text[:remaining]
        if not text.strip():
            continue
        sources.append(
            AnswerSource(
                source_id=f"S{index}",
                chunk_id=result.chunk_id,
                source_type=result.source_type,
                title=result.title,
                source_url=result.source_url,
                repository=result.repository,
                text=text,
                metadata=result.metadata,
            )
        )
        remaining -= len(text)
    records = [
        {
            "source_id": source.source_id,
            "source_type": source.source_type.value,
            "title": source.title,
            "github_url": str(source.source_url),
            "text": source.text,
        }
        for source in sources
    ]
    payload = json.dumps(records, ensure_ascii=False, indent=2)
    return (
        f'<retrieved_evidence trust="untrusted-data">\n{payload}\n</retrieved_evidence>',
        sources,
    )
