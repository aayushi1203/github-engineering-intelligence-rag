import re
import uuid
from collections.abc import Iterable

from github_intelligence.models import DocumentChunk, DocumentType, GitHubDocument

_HEADING = re.compile(r"^(#{1,6}\s+.+|[^\n]+\n(?:={3,}|-{3,}))$", re.MULTILINE)
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`])")


class DocumentChunker:
    """Structure-aware chunker for GitHub conversations and documentation."""

    def __init__(self, max_words: int = 220, overlap_words: int = 30) -> None:
        if max_words < 20:
            raise ValueError("max_words must be at least 20")
        if overlap_words < 0 or overlap_words >= max_words:
            raise ValueError("overlap_words must be between 0 and max_words")
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk_documents(self, documents: Iterable[GitHubDocument]) -> list[DocumentChunk]:
        return [chunk for document in documents for chunk in self.chunk(document)]

    def chunk(self, document: GitHubDocument) -> list[DocumentChunk]:
        pieces: list[tuple[str | None, str]] = []
        for heading, body in self._sections(document):
            pieces.extend((heading, part) for part in self._bounded_parts(body))
        if not pieces:
            pieces = [(None, document.text)]

        chunks: list[DocumentChunk] = []
        for index, (heading, body) in enumerate(pieces):
            context = [document.title]
            if heading and heading.strip().lower() != document.title.strip().lower():
                context.append(heading.strip())
            context.append(body.strip())
            text = "\n\n".join(dict.fromkeys(part for part in context if part))
            chunk_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL, f"{document.repository}:{document.github_id}:{index}"
                )
            )
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "github_id": document.github_id,
                    "chunk_index": index,
                    "chunk_count": len(pieces),
                    "section_heading": heading,
                }
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.github_id,
                    chunk_index=index,
                    text=text,
                    title=document.title,
                    source_url=document.source_url,
                    source_type=document.document_type,
                    repository=document.repository,
                    metadata=metadata,
                )
            )
        return chunks

    def _sections(self, document: GitHubDocument) -> list[tuple[str | None, str]]:
        if document.document_type != DocumentType.DOCUMENTATION:
            body = document.text
            if body.startswith(document.title):
                remainder = body[len(document.title) :].lstrip(" \n:-")
                if remainder:
                    body = remainder
            return [(None, body)]
        matches = list(_HEADING.finditer(document.text))
        if not matches:
            return [(None, document.text)]
        sections: list[tuple[str | None, str]] = []
        before_first = document.text[: matches[0].start()].strip()
        if before_first:
            sections.append((None, before_first))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(document.text)
            heading = match.group(0).replace("\n", " ").lstrip("# ").strip(" =-")
            body = document.text[match.end() : end].strip()
            if body:
                sections.append((heading, body))
        return sections

    def _bounded_parts(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in _PARAGRAPH_BREAK.split(text) if part.strip()]
        units: list[str] = []
        for paragraph in paragraphs or [text]:
            if len(paragraph.split()) <= self.max_words:
                units.append(paragraph)
            else:
                sentences = [
                    part.strip() for part in _SENTENCE_BREAK.split(paragraph) if part.strip()
                ]
                for sentence in sentences:
                    units.extend(self._word_windows(sentence))
        return self._pack(units)

    def _word_windows(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.max_words:
            return [text]
        step = self.max_words - self.overlap_words
        return [
            " ".join(words[start : start + self.max_words])
            for start in range(0, len(words), step)
        ]

    def _pack(self, units: list[str]) -> list[str]:
        packed: list[str] = []
        current: list[str] = []
        current_words = 0
        for unit in units:
            unit_words = len(unit.split())
            if current and current_words + unit_words > self.max_words:
                packed.append("\n\n".join(current))
                current, current_words = [], 0
            if unit_words > self.max_words:
                packed.extend(self._word_windows(unit))
            else:
                current.append(unit)
                current_words += unit_words
        if current:
            packed.append("\n\n".join(current))
        return packed
