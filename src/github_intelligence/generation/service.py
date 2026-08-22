import re

from github_intelligence.generation.context import build_context
from github_intelligence.generation.models import Citation, DraftAnswer, GroundedAnswer
from github_intelligence.generation.prompts import SYSTEM_PROMPT, repair_prompt, user_prompt
from github_intelligence.generation.provider import LLMProvider
from github_intelligence.models import RetrievalResult
from github_intelligence.retrieval import RetrievalMode, RetrievalPipeline

CITATION_PATTERN = re.compile(r"\[(S[1-9]\d*)\]")


class CitationValidationError(ValueError):
    pass


def parse_citation_ids(answer: str) -> list[str]:
    return list(dict.fromkeys(CITATION_PATTERN.findall(answer)))


def validate_draft(draft: DraftAnswer, valid_ids: set[str]) -> list[str]:
    inline_ids = parse_citation_ids(draft.answer)
    declared_ids = list(dict.fromkeys(draft.citation_ids))
    invalid = (set(inline_ids) | set(declared_ids)) - valid_ids
    if invalid:
        raise CitationValidationError(f"nonexistent citation IDs: {', '.join(sorted(invalid))}")
    if set(inline_ids) != set(declared_ids):
        raise CitationValidationError("inline citations and citation_ids do not match")
    if not inline_ids and not draft.limitations:
        raise CitationValidationError(
            "an answer without citations must explicitly report a limitation"
        )
    return inline_ids


class GroundedAnswerService:
    def __init__(
        self,
        pipeline: RetrievalPipeline,
        provider: LLMProvider,
        *,
        top_k: int = 8,
        max_context_chars: int = 30_000,
        repair_attempts: int = 1,
    ) -> None:
        self.pipeline = pipeline
        self.provider = provider
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.repair_attempts = repair_attempts

    def ask(self, question: str, *, repository: str) -> GroundedAnswer:
        results = self.pipeline.search(
            question, mode=RetrievalMode.HYBRID_RERANK, top_k=self.top_k, repository=repository
        )
        return self.answer_from_results(question, repository=repository, results=results)

    def answer_from_results(
        self,
        question: str,
        *,
        repository: str,
        results: list[RetrievalResult],
    ) -> GroundedAnswer:
        """Generate from an explicit evidence set, including source-compatible MCP evidence."""
        context, sources = build_context(results, max_chars=self.max_context_chars)
        if not sources:
            return GroundedAnswer(
                answer="The retrieved evidence is insufficient to answer this question.",
                citations=[],
                sources=[],
                retrieved_sources=[],
                limitations=["No relevant indexed GitHub evidence was retrieved."],
            )
        prompt = user_prompt(question, repository, context)
        valid_ids = {source.source_id for source in sources}
        last_error: CitationValidationError | None = None
        for attempt in range(self.repair_attempts + 1):
            draft = self.provider.generate(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
            try:
                cited_ids = validate_draft(draft, valid_ids)
                by_id = {source.source_id: source for source in sources}
                cited_sources = [by_id[source_id] for source_id in cited_ids]
                citations = [
                    Citation(source_id=s.source_id, source_url=s.source_url) for s in cited_sources
                ]
                return GroundedAnswer(
                    answer=draft.answer,
                    citations=citations,
                    sources=cited_sources,
                    retrieved_sources=sources,
                    limitations=draft.limitations or None,
                )
            except CitationValidationError as exc:
                last_error = exc
                if attempt < self.repair_attempts:
                    prompt = f"{prompt}\n\n{repair_prompt(str(exc), valid_ids)}"
        raise last_error or CitationValidationError("citation validation failed")
