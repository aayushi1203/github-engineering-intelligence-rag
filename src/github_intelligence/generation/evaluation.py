import re

from github_intelligence.generation.models import GenerationMetrics, GroundedAnswer
from github_intelligence.generation.service import parse_citation_ids

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]{3,}")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
CITATION_STRIP = re.compile(r"\[S\d+\]")
STOPWORDS = {"the", "and", "that", "this", "with", "from", "for", "was", "are", "were"}


def evaluate_grounded_answer(response: GroundedAnswer) -> GenerationMetrics:
    """Deterministic proxies; groundedness is token support, not an LLM-judged score."""
    valid_ids = {source.source_id for source in response.retrieved_sources}
    answer_ids = parse_citation_ids(response.answer)
    validity = sum(sid in valid_ids for sid in answer_ids) / len(answer_ids) if answer_ids else 1.0
    claims = [
        s.strip()
        for s in SENTENCE_PATTERN.split(response.answer)
        if len(TOKEN_PATTERN.findall(s)) >= 4
    ]
    cited_claims = [claim for claim in claims if parse_citation_ids(claim)]
    coverage = len(cited_claims) / len(claims) if claims else 1.0
    sources = {source.source_id: source.text for source in response.retrieved_sources}
    scores: list[float] = []
    for claim in cited_claims:
        evidence_tokens = set(
            _tokens(" ".join(sources.get(sid, "") for sid in parse_citation_ids(claim)))
        )
        claim_tokens = set(_tokens(CITATION_STRIP.sub("", claim)))
        scores.append(
            len(claim_tokens & evidence_tokens) / len(claim_tokens) if claim_tokens else 1.0
        )
    return GenerationMetrics(
        citation_validity=validity,
        citation_coverage=coverage,
        groundedness=sum(scores) / len(scores) if scores else 0.0,
        claim_count=len(claims),
        cited_claim_count=len(cited_claims),
    )


def _tokens(text: str) -> list[str]:
    return [
        token.lower() for token in TOKEN_PATTERN.findall(text) if token.lower() not in STOPWORDS
    ]
