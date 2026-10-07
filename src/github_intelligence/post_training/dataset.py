import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from github_intelligence.evaluation.models import EvaluationExample
from github_intelligence.models import DocumentType, GitHubDocument
from github_intelligence.post_training.models import (
    DatasetSplit,
    FeasibilitySummary,
    GoldEvidence,
    PolicyExample,
)


class SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train: list[str] = Field(min_length=1)
    development: list[str] = Field(min_length=1)
    test: list[str] = Field(min_length=1)

    def split_for(self, repository: str) -> DatasetSplit:
        matches = [
            split
            for split, repositories in (
                (DatasetSplit.TRAIN, self.train),
                (DatasetSplit.DEVELOPMENT, self.development),
                (DatasetSplit.TEST, self.test),
            )
            if repository in repositories
        ]
        if len(matches) != 1:
            raise ValueError(f"Repository {repository!r} must occur in exactly one split")
        return matches[0]

    @property
    def repositories(self) -> set[str]:
        groups = self.train + self.development + self.test
        if len(groups) != len(set(groups)):
            raise ValueError("Repository splits must be disjoint")
        return set(groups)


def load_policy_dataset(path: Path) -> list[PolicyExample]:
    examples: list[PolicyExample] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                examples.append(PolicyExample.model_validate_json(line))
            except ValueError as error:
                message = f"Invalid policy example on line {line_number}: {error}"
                raise ValueError(message) from error
    return examples


def write_policy_dataset(path: Path, examples: Iterable[PolicyExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(example.model_dump_json() + "\n" for example in examples),
        encoding="utf-8",
    )


def load_manifest(path: Path) -> SplitManifest:
    return SplitManifest.model_validate_json(path.read_text(encoding="utf-8"))


def build_policy_dataset(
    evaluations: Iterable[EvaluationExample],
    documents: Iterable[GitHubDocument],
    manifest: SplitManifest,
) -> list[PolicyExample]:
    _ = manifest.repositories
    document_index: dict[tuple[str, str], GitHubDocument] = {
        (document.repository, document.github_id): document for document in documents
    }
    examples: list[PolicyExample] = []
    for evaluation in evaluations:
        split = manifest.split_for(evaluation.repository)
        gold: list[GoldEvidence] = []
        for source_id in evaluation.relevant_source_ids:
            document = document_index.get((evaluation.repository, source_id))
            if document is None:
                raise ValueError(
                    f"Missing relevance label {source_id} in corpus for {evaluation.repository}"
                )
            gold.append(
                GoldEvidence(
                    source_id=source_id,
                    source_url=document.source_url,
                    source_type=document.document_type,
                    title=document.title,
                )
            )
        digest = hashlib.sha256(
            f"{evaluation.repository}\0{evaluation.query}".encode()
        ).hexdigest()[:16]
        examples.append(
            PolicyExample(
                example_id=f"policy-{digest}",
                repository=evaluation.repository,
                question=evaluation.query,
                split=split,
                gold_evidence=gold,
                acceptable_tools=_acceptable_tools(gold),
                notes=evaluation.notes,
            )
        )
    return examples


def build_corpus_tool_examples(
    documents: Iterable[GitHubDocument],
    manifest: SplitManifest,
    *,
    limit: int,
    split: DatasetSplit = DatasetSplit.TRAIN,
    excluded_sources: set[tuple[str, str]] | None = None,
) -> list[PolicyExample]:
    """Create tool-routing examples backed by real, split-isolated corpus artifacts."""

    if limit < 0:
        raise ValueError("Corpus augmentation limit cannot be negative")
    candidates = sorted(
        (
            document
            for document in documents
            if manifest.split_for(document.repository) == split
            and (document.repository, document.github_id) not in (excluded_sources or set())
        ),
        key=lambda item: (item.repository, item.document_type.value, item.github_id),
    )
    examples: list[PolicyExample] = []
    for document in _balanced_by_tool(candidates, limit):
        question = f"Find the most relevant GitHub evidence about: {document.title}"
        digest = hashlib.sha256(
            f"corpus\0{document.repository}\0{document.github_id}".encode()
        ).hexdigest()[:16]
        gold = GoldEvidence(
            source_id=document.github_id,
            source_url=document.source_url,
            source_type=document.document_type,
            title=document.title,
        )
        examples.append(
            PolicyExample(
                example_id=f"corpus-{digest}",
                repository=document.repository,
                question=question,
                split=split,
                gold_evidence=[gold],
                acceptable_tools=_acceptable_tools([gold]),
                notes=(
                    f"{split.value.title()} routing example derived from a "
                    "provenance-checked corpus source."
                ),
            )
        )
    return examples


def summarize_dataset(examples: list[PolicyExample]) -> FeasibilitySummary:
    repositories = {
        split: sorted({item.repository for item in examples if item.split == split})
        for split in DatasetSplit
    }
    split_counts = {
        split: sum(item.split == split for item in examples) for split in DatasetSplit
    }
    labeled = sum(bool(item.gold_evidence) for item in examples)
    return FeasibilitySummary(
        examples=len(examples),
        repositories=repositories,
        split_counts=split_counts,
        label_coverage=labeled / len(examples) if examples else 0.0,
    )


def load_evaluations(path: Path) -> list[EvaluationExample]:
    result: list[EvaluationExample] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                result.append(EvaluationExample.model_validate_json(line))
    return result


def load_corpora(paths: Iterable[Path]) -> list[GitHubDocument]:
    documents: list[GitHubDocument] = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    documents.append(GitHubDocument.model_validate_json(line))
    return documents


def _acceptable_tools(gold: list[GoldEvidence]) -> list[str]:
    mapping: Mapping[DocumentType, str] = {
        DocumentType.ISSUE: "search_issues",
        DocumentType.ISSUE_COMMENT: "search_issues",
        DocumentType.PULL_REQUEST: "search_pull_requests",
        DocumentType.PULL_REQUEST_COMMENT: "search_pull_requests",
        DocumentType.PULL_REQUEST_REVIEW_COMMENT: "search_pull_requests",
        DocumentType.COMMIT: "search_repository_documents",
        DocumentType.RELEASE: "search_repository_documents",
        DocumentType.DOCUMENTATION: "search_repository_documents",
    }
    return list(dict.fromkeys(mapping[item.source_type] for item in gold))


def _balanced_by_tool(documents: list[GitHubDocument], limit: int) -> list[GitHubDocument]:
    groups: dict[str, list[GitHubDocument]] = {}
    for document in documents:
        evidence = GoldEvidence(
            source_id=document.github_id,
            source_url=document.source_url,
            source_type=document.document_type,
            title=document.title,
        )
        tool = _acceptable_tools([evidence])[0]
        groups.setdefault(tool, []).append(document)
    selected: list[GitHubDocument] = []
    offsets = {tool: 0 for tool in groups}
    while len(selected) < limit:
        added = False
        for tool in sorted(groups):
            offset = offsets[tool]
            if offset < len(groups[tool]) and len(selected) < limit:
                selected.append(groups[tool][offset])
                offsets[tool] += 1
                added = True
        if not added:
            break
    return selected


def manifest_json(manifest: SplitManifest) -> str:
    return json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n"
