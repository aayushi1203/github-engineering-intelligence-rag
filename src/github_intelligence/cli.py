import argparse
import asyncio
import sys
from pathlib import Path

from github_intelligence.agent import InvestigationAgent
from github_intelligence.config import Settings
from github_intelligence.evaluation import RetrievalEvaluator, load_evaluation_dataset
from github_intelligence.generation import (
    GenerationEvaluator,
    GroundedAnswerService,
    OpenAIResponsesProvider,
    load_generation_dataset,
)
from github_intelligence.github import GitHubClient
from github_intelligence.ingestion import IngestionOptions, RepositoryIngestor
from github_intelligence.logging import configure_logging
from github_intelligence.mcp import GitHubEngineeringTools
from github_intelligence.models import GitHubDocument, RetrievalResult
from github_intelligence.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DocumentChunker,
    QdrantVectorRetriever,
    QdrantVectorStore,
    RetrievalMode,
    RetrievalPipeline,
    SentenceTransformerEncoder,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GitHub engineering intelligence toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("repository", help="GitHub repository in owner/name format")
    ingest.add_argument("--output", type=Path, required=True, help="Destination JSONL file")
    ingest.add_argument("--no-comments", action="store_true")
    ingest.add_argument("--no-docs", action="store_true")
    ingest.add_argument("--log-level", default=None)

    index = subparsers.add_parser("index", help="Chunk and index an ingestion JSONL file")
    index.add_argument("input", type=Path, help="Phase 1 JSONL file")
    index.add_argument("--backend", choices=("both", "bm25", "vector"), default="both")
    index.add_argument("--log-level", default=None)

    search = subparsers.add_parser("search", help="Search an existing retrieval index")
    search.add_argument("query")
    search.add_argument("--repo", default=None, help="Restrict results to owner/name")
    search.add_argument(
        "--backend",
        choices=tuple(mode.value for mode in RetrievalMode),
        default=RetrievalMode.HYBRID_RERANK.value,
    )
    search.add_argument("--top-k", type=int, default=None)
    search.add_argument("--log-level", default=None)

    evaluate = subparsers.add_parser(
        "evaluate-retrieval", help="Compare retrieval modes on a labeled JSONL benchmark"
    )
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--repo", default=None, help="Evaluate only one owner/name repository")
    evaluate.add_argument("--top-k", type=int, default=10, help="Retrieval depth (minimum 10)")
    evaluate.add_argument("--output", type=Path, default=Path("output/retrieval-evaluation.json"))
    evaluate.add_argument("--log-level", default=None)

    evaluate_generation = subparsers.add_parser(
        "evaluate-generation", help="Evaluate generated answers with deterministic metrics"
    )
    evaluate_generation.add_argument("--dataset", type=Path, required=True)
    evaluate_generation.add_argument(
        "--repo", default=None, help="Evaluate only one owner/name repository"
    )
    evaluate_generation.add_argument(
        "--output", type=Path, default=Path("output/generation-evaluation.json")
    )
    evaluate_generation.add_argument("--top-k", type=int, default=None)
    evaluate_generation.add_argument("--log-level", default=None)

    ask = subparsers.add_parser("ask", help="Generate a cited answer from retrieved evidence")
    ask.add_argument("question")
    ask.add_argument("--repo", required=True, help="Repository in owner/name format")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--log-level", default=None)

    investigate = subparsers.add_parser(
        "investigate", help="Run a bounded GitHub tool-assisted engineering investigation"
    )
    investigate.add_argument("question")
    investigate.add_argument("--repo", required=True, help="Repository in owner/name format")
    investigate.add_argument("--log-level", default=None)
    return parser


async def run_ingest(args: argparse.Namespace, settings: Settings) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    async with GitHubClient(settings) as client:
        ingestor = RepositoryIngestor(client, settings.github_doc_max_bytes)
        options = IngestionOptions(not args.no_comments, not args.no_docs)
        with args.output.open("w", encoding="utf-8") as output:
            async for document in ingestor.ingest(args.repository, options):
                output.write(document.model_dump_json() + "\n")
                count += 1
    print(f"Wrote {count} documents to {args.output}")
    return 0


def load_documents(path: Path) -> list[GitHubDocument]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    with path.open(encoding="utf-8") as source:
        return [GitHubDocument.model_validate_json(line) for line in source if line.strip()]


def vector_retriever(settings: Settings) -> QdrantVectorRetriever:
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantVectorRetriever(
        QdrantVectorStore(settings.qdrant_path, settings.qdrant_collection),
        SentenceTransformerEncoder(settings.embedding_model, settings.embedding_device),
    )


def run_index(args: argparse.Namespace, settings: Settings) -> int:
    documents = load_documents(args.input)
    chunker = DocumentChunker(settings.chunk_max_words, settings.validated_chunk_overlap_words)
    chunks = chunker.chunk_documents(documents)
    if args.backend in {"both", "bm25"}:
        BM25Retriever(settings.index_dir / "chunks.jsonl").index(chunks)
    if args.backend in {"both", "vector"}:
        vector_retriever(settings).index(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents ({args.backend})")
    return 0


def run_search(args: argparse.Namespace, settings: Settings) -> int:
    top_k = args.top_k or settings.retrieval_top_k
    if top_k < 1:
        raise ValueError("top-k must be at least 1")
    mode = RetrievalMode(args.backend)
    pipeline = build_retrieval_pipeline(settings, {mode})
    results = pipeline.search(args.query, mode=mode, top_k=top_k, repository=args.repo)
    display_results(results)
    return 0


def build_retrieval_pipeline(settings: Settings, modes: set[RetrievalMode]) -> RetrievalPipeline:
    needs_bm25 = bool(
        modes & {RetrievalMode.BM25, RetrievalMode.HYBRID, RetrievalMode.HYBRID_RERANK}
    )
    needs_vector = bool(
        modes & {RetrievalMode.VECTOR, RetrievalMode.HYBRID, RetrievalMode.HYBRID_RERANK}
    )
    needs_reranker = RetrievalMode.HYBRID_RERANK in modes
    return RetrievalPipeline(
        bm25=(BM25Retriever(settings.index_dir / "chunks.jsonl") if needs_bm25 else None),
        vector=(vector_retriever(settings) if needs_vector else None),
        reranker=(
            CrossEncoderReranker(settings.reranker_model, settings.reranker_device)
            if needs_reranker
            else None
        ),
        candidates_per_backend=settings.retrieval_candidates_per_backend,
        rerank_candidate_pool=settings.rerank_candidate_pool,
        rrf_k=settings.rrf_k,
    )


def run_evaluation(args: argparse.Namespace, settings: Settings) -> int:
    examples = load_evaluation_dataset(args.dataset)
    if args.repo:
        examples = [example for example in examples if example.repository == args.repo]
        if not examples:
            raise ValueError(f"Dataset has no examples for repository {args.repo}")
    modes = tuple(RetrievalMode)
    pipeline = build_retrieval_pipeline(settings, set(modes))
    report = RetrievalEvaluator(pipeline, settings).evaluate(
        examples,
        dataset_path=args.dataset,
        modes=modes,
        retrieval_depth=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print("| Backend | Recall@5 | Recall@10 | MRR | NDCG@10 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for mode in modes:
        metric = report.metrics[mode]
        print(
            f"| {mode.value} | {metric.recall_at_5:.4f} | {metric.recall_at_10:.4f} "
            f"| {metric.mrr:.4f} | {metric.ndcg_at_10:.4f} |"
        )
    print(f"\nSaved diagnostics to {args.output}")
    return 0


def display_results(results: list[RetrievalResult]) -> None:
    if not results:
        print("No matching chunks found.")
        return
    for rank, result in enumerate(results, start=1):
        preview = " ".join(result.text.split())
        if len(preview) > 300:
            preview = preview[:297] + "..."
        display_rank = result.final_rank or rank
        print(f"{display_rank}. [{result.source_type.value}] {result.title}")
        scores = [f"final={result.score:.4f}"]
        for label, value in (
            ("bm25", result.bm25_score),
            ("vector", result.vector_score),
            ("rrf", result.rrf_score),
            ("reranker", result.reranker_score),
        ):
            if value is not None:
                scores.append(f"{label}={value:.4f}")
        ranks = []
        if result.bm25_rank is not None:
            ranks.append(f"bm25_rank={result.bm25_rank}")
        if result.vector_rank is not None:
            ranks.append(f"vector_rank={result.vector_rank}")
        detail = " ".join(scores + ranks)
        print(f"   {detail} repo={result.repository}")
        print(f"   {result.source_url}")
        print(f"   {preview}")


def run_ask(args: argparse.Namespace, settings: Settings) -> int:
    try:
        provider = OpenAIResponsesProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            api_url=settings.openai_api_url,
            max_output_tokens=settings.generation_max_output_tokens,
        )
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    service = GroundedAnswerService(
        build_retrieval_pipeline(settings, {RetrievalMode.HYBRID_RERANK}),
        provider,
        top_k=args.top_k or settings.generation_top_k,
        max_context_chars=settings.generation_max_context_chars,
        repair_attempts=settings.generation_citation_repair_attempts,
    )
    response = service.ask(args.question, repository=args.repo)
    print(response.answer)
    if response.sources:
        print("\nSources")
        for source in response.sources:
            print(f"- [{source.source_id}] {source.title} ({source.source_url})")
    if response.limitations:
        print("\nLimitations")
        for limitation in response.limitations:
            print(f"- {limitation}")
    return 0


async def run_investigate(args: argparse.Namespace, settings: Settings) -> int:
    try:
        provider = OpenAIResponsesProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            api_url=settings.openai_api_url,
            max_output_tokens=settings.generation_max_output_tokens,
        )
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    pipeline = build_retrieval_pipeline(settings, {RetrievalMode.HYBRID_RERANK})
    answer_service = GroundedAnswerService(
        pipeline,
        provider,
        top_k=settings.generation_top_k,
        max_context_chars=settings.generation_max_context_chars,
        repair_attempts=settings.generation_citation_repair_attempts,
    )
    async with GitHubClient(settings) as client:
        agent = InvestigationAgent(
            pipeline,
            answer_service,
            GitHubEngineeringTools(client, pipeline),
            max_steps=settings.investigation_max_steps,
            max_mcp_calls=settings.investigation_max_mcp_calls,
            max_sources=settings.investigation_max_sources,
            search_limit=settings.mcp_search_limit,
        )
        result = await agent.investigate(args.question, repository=args.repo)
    print(result.answer.answer)
    print("\nInvestigation steps")
    for item in result.investigation_steps:
        print(f"- {item.step}. {item.detail}")
    if result.tool_calls:
        print("\nMCP/tool calls")
        for call in result.tool_calls:
            status = "ok" if call.succeeded else "failed"
            print(f"- {call.tool_name}: {status}, evidence={call.evidence_count}")
    if result.answer.sources:
        print("\nSources")
        for source in result.answer.sources:
            print(f"- [{source.source_id}] {source.title} ({source.source_url})")
    if result.limitations:
        print("\nLimitations")
        for limitation in result.limitations:
            print(f"- {limitation}")
    return 0


def run_generation_evaluation(args: argparse.Namespace, settings: Settings) -> int:
    examples = load_generation_dataset(args.dataset, repository=args.repo)
    top_k = args.top_k or settings.generation_top_k
    if top_k < 1:
        raise ValueError("top-k must be at least 1")
    try:
        provider = OpenAIResponsesProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            api_url=settings.openai_api_url,
            max_output_tokens=settings.generation_max_output_tokens,
        )
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    service = GroundedAnswerService(
        build_retrieval_pipeline(settings, {RetrievalMode.HYBRID_RERANK}),
        provider,
        top_k=top_k,
        max_context_chars=settings.generation_max_context_chars,
        repair_attempts=settings.generation_citation_repair_attempts,
    )
    report = GenerationEvaluator(service).evaluate(examples, dataset_path=args.dataset, top_k=top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    aggregate = report.aggregate
    print(f"Examples evaluated: {aggregate.examples_evaluated}")
    print(f"Successful generations: {aggregate.successful_generations}")
    print(f"Mean citation validity: {_format_optional_metric(aggregate.mean_citation_validity)}")
    print(f"Mean citation coverage: {_format_optional_metric(aggregate.mean_citation_coverage)}")
    print(f"Mean groundedness: {_format_optional_metric(aggregate.mean_groundedness)}")
    print(f"Total claims: {aggregate.total_claim_count}")
    print(f"Cited claims: {aggregate.total_cited_claim_count}")
    print(f"Output: {args.output}")
    return 0


def _format_optional_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "N/A"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings()
    configure_logging(args.log_level or settings.log_level)
    if args.command == "ingest":
        raise SystemExit(asyncio.run(run_ingest(args, settings)))
    if args.command == "index":
        raise SystemExit(run_index(args, settings))
    if args.command == "search":
        raise SystemExit(run_search(args, settings))
    if args.command == "evaluate-retrieval":
        raise SystemExit(run_evaluation(args, settings))
    if args.command == "evaluate-generation":
        raise SystemExit(run_generation_evaluation(args, settings))
    if args.command == "ask":
        raise SystemExit(run_ask(args, settings))
    if args.command == "investigate":
        raise SystemExit(asyncio.run(run_investigate(args, settings)))


if __name__ == "__main__":
    main()
