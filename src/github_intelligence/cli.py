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
from github_intelligence.post_training import (
    build_corpus_tool_examples,
    build_policy_dataset,
    evaluate_policy_checkpoint,
    load_corpora,
    load_evaluations,
    load_manifest,
    summarize_dataset,
    write_policy_dataset,
)
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
    serve = subparsers.add_parser("serve", help="Run the trusted-local HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--log-level", default=None)
    worker = subparsers.add_parser("worker", help="Run the PostgreSQL-backed ingestion worker")
    worker.add_argument("--once", action="store_true", help="Claim at most one job then exit")
    worker.add_argument("--log-level", default=None)
    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("repository", help="GitHub repository in owner/name format")
    ingest.add_argument("--output", type=Path, required=True, help="Destination JSONL file")
    ingest.add_argument("--no-comments", action="store_true")
    ingest.add_argument("--no-docs", action="store_true")
    ingest.add_argument("--no-reviews", action="store_true")
    ingest.add_argument(
        "--cached", action="store_true", help="Permit cached reads; default is fresh"
    )
    ingest.add_argument(
        "--allow-incomplete", action="store_true",
        help="Publish explicitly marked limited/truncated scans; never failed scans",
    )
    ingest.add_argument("--log-level", default=None)

    metadata = subparsers.add_parser(
        "import-metadata", help="Persist JSONL metadata without changing retrieval indexes"
    )
    metadata.add_argument("input", type=Path)
    metadata.add_argument("--log-level", default=None)

    incremental = subparsers.add_parser(
        "index-incremental", help="Index a validated scan into isolated application stores"
    )
    incremental.add_argument("input", type=Path)
    incremental.add_argument("--repo", required=True)
    incremental.add_argument("--root", type=Path, default=Path(".cache/application-index"))
    incremental.add_argument("--embedding-version", required=True,
                             help="Explicit immutable model/version identifier")
    incremental.add_argument("--reconcile", action="store_true",
                             help="Permit only authoritative, exact-scope documentation deletion")
    incremental.add_argument("--qdrant-mode", choices=("embedded", "server"), default="embedded")
    incremental.add_argument("--log-level", default=None)
    retire = subparsers.add_parser(
        "retire-index", help="Retire a non-current revision after its readers have drained"
    )
    retire.add_argument("revision", help="Indexing operation UUID")
    retire.add_argument("--repo", required=True)
    retire.add_argument("--root", type=Path, default=Path(".cache/application-index"))
    retire.add_argument("--embedding-version", required=True)
    retire.add_argument("--qdrant-mode", choices=("embedded", "server"), default="embedded")
    retire.add_argument("--log-level", default=None)

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

    prepare_policy = subparsers.add_parser(
        "prepare-policy-data",
        help="Build a provenance-checked, repository-disjoint tool-policy dataset",
    )
    prepare_policy.add_argument("--evaluation", type=Path, required=True)
    prepare_policy.add_argument("--corpus", type=Path, action="append", required=True)
    prepare_policy.add_argument("--splits", type=Path, required=True)
    prepare_policy.add_argument(
        "--output", type=Path, default=Path("data/post_training/policy.jsonl")
    )
    prepare_policy.add_argument(
        "--augment-training-from-corpus",
        type=int,
        default=0,
        metavar="N",
        help="Add at most N provenance-backed routing examples from training repositories",
    )
    prepare_policy.add_argument(
        "--augment-test-from-corpus",
        type=int,
        default=0,
        metavar="N",
        help="Add at most N provenance-backed routing examples from held-out repositories",
    )
    prepare_policy.add_argument("--log-level", default=None)

    train_policy = subparsers.add_parser(
        "train-policy", help="Run the isolated QLoRA SFT or GRPO tool-policy experiment"
    )
    train_policy.add_argument("--dataset", type=Path, required=True)
    train_policy.add_argument("--stage", choices=("sft", "grpo"), required=True)
    train_policy.add_argument("--model", default="Qwen/Qwen3-4B")
    train_policy.add_argument("--output", type=Path, required=True)
    train_policy.add_argument("--max-steps", type=int, default=None)
    train_policy.add_argument("--seed", type=int, default=42)
    train_policy.add_argument("--num-generations", type=int, default=4)
    train_policy.add_argument("--log-level", default=None)

    evaluate_policy = subparsers.add_parser(
        "evaluate-policy", help="Evaluate one policy checkpoint with deterministic metrics"
    )
    evaluate_policy.add_argument("--dataset", type=Path, required=True)
    evaluate_policy.add_argument("--model", required=True)
    evaluate_policy.add_argument("--output", type=Path, required=True)
    evaluate_policy.add_argument(
        "--split", choices=("development", "test"), default="test"
    )
    evaluate_policy.add_argument("--max-new-tokens", type=int, default=192)
    evaluate_policy.add_argument("--log-level", default=None)
    return parser


async def run_ingest(args: argparse.Namespace, settings: Settings) -> int:
    from github_intelligence.ingestion.service import IngestionError
    from github_intelligence.ingestion.snapshot import publish_snapshot, report_path

    async with GitHubClient(settings) as client:
        ingestor = RepositoryIngestor(client, settings.github_doc_max_bytes)
        options = IngestionOptions(not args.no_comments, not args.no_docs,
                                   not args.no_reviews, not args.cached)
        try:
            report = await publish_snapshot(ingestor, args.repository, args.output, options,
                                            allow_incomplete=args.allow_incomplete)
        except (IngestionError, OSError, ValueError) as error:
            code = (
                error.code if isinstance(error, IngestionError)
                else "invalid_input_or_publication_failure"
            )
            print(
                f"Ingestion failed: {code}. No successful snapshot was published.", file=sys.stderr
            )
            return 1
    count = sum(source.documents_emitted for source in report.sources.values())
    scope = "full" if report.full_snapshot else "scoped/partial"
    print(f"Published {count} documents ({scope}) to {args.output}")
    print(f"Scan metadata: {report_path(args.output)}")
    return 0


def load_documents(path: Path) -> list[GitHubDocument]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    with path.open(encoding="utf-8") as source:
        return [GitHubDocument.model_validate_json(line) for line in source if line.strip()]


def run_import_metadata(args: argparse.Namespace, settings: Settings) -> int:
    # Optional dependencies load only for this explicitly requested operation.
    if not settings.database_url:
        print("DATABASE_URL is required for import-metadata.", file=sys.stderr)
        return 1
    try:
        from sqlalchemy.exc import SQLAlchemyError

        from github_intelligence.persistence.database import (
            PersistenceError,
            create_database_engine,
            session_factory,
        )
        from github_intelligence.persistence.operations import import_documents
    except ImportError:
        print(
            "Install metadata dependencies with: pip install -e '.[persistence]'", file=sys.stderr
        )
        return 1
    try:
        documents = load_documents(args.input)
    except (OSError, ValueError):
        print("Metadata input must be a valid GitHubDocument JSONL file.", file=sys.stderr)
        return 1
    try:
        engine = create_database_engine(settings.database_url)
        try:
            with session_factory(engine).begin() as session:
                counts = import_documents(session, documents)
        finally:
            engine.dispose()
    except (PersistenceError, SQLAlchemyError, ValueError):
        # Database errors may contain credentials, bound values, or user content.
        print(
            "Metadata import failed; verify input identities, PostgreSQL connectivity, "
            "and `alembic upgrade head`. No changes were committed.",
            file=sys.stderr,
        )
        return 1
    print(f"Metadata: new={counts.new} changed={counts.changed} unchanged={counts.unchanged}")
    print("New/changed documents are pending; retrieval indexes were not modified.")
    return 0


def vector_retriever(settings: Settings) -> QdrantVectorRetriever:
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantVectorRetriever(
        QdrantVectorStore(settings.qdrant_path, settings.qdrant_collection),
        SentenceTransformerEncoder(settings.embedding_model, settings.embedding_device),
    )


def run_incremental(args: argparse.Namespace, settings: Settings) -> int:
    from uuid import UUID

    try:
        from sqlalchemy.exc import SQLAlchemyError

        from github_intelligence.indexing.coordinator import IncrementalCoordinator, IndexingError
        from github_intelligence.indexing.storage import IncrementalVectorStore
        from github_intelligence.persistence.database import (
            PersistenceError,
            create_database_engine,
        )
    except ImportError:
        print("Incremental indexing requires the persistence extra", file=sys.stderr)
        return 2
    if args.root.resolve() in {settings.index_dir.resolve(), settings.qdrant_path.resolve()}:
        print("Application root must be separate from offline benchmark stores", file=sys.stderr)
        return 2
    engine = None
    store = None
    try:
        engine = create_database_engine(settings.database_url)
        store = IncrementalVectorStore(
            args.root / "qdrant" if args.qdrant_mode == "embedded" else None,
            "application_chunks",
            url=settings.application_qdrant_url if args.qdrant_mode == "server" else None,
            api_key=settings.application_qdrant_api_key,
            timeout=settings.application_qdrant_timeout_seconds,
        )
        coordinator = IncrementalCoordinator(
            engine, store,
            SentenceTransformerEncoder(settings.embedding_model, settings.embedding_device),
            args.root, embedding_version=f"{settings.embedding_model}@{args.embedding_version}",
            chunker=DocumentChunker(
                settings.chunk_max_words, settings.validated_chunk_overlap_words
            ),
        )
        if args.command == "retire-index":
            coordinator.retire_revision(UUID(args.revision), repository=args.repo)
            print(f"Retired revision {args.revision}; current READY preserved")
        else:
            operation = coordinator.run(args.input, repository=args.repo, reconcile=args.reconcile)
            assert operation.plan is not None
            print(" ".join(f"{key}={value}" for key, value in operation.plan["counts"].items()))
            print(f"revision={operation.id} status={operation.status}")
            incomplete = {family: scan["status"]
                          for family, scan in operation.scan_report["sources"].items()
                          if scan["status"] != "complete"}
            if incomplete:
                print(f"Incomplete/excluded scan families: {incomplete}")
            for identity, reason in operation.plan["preserved"].items():
                print(f"Preserved {identity}: {reason}")
        return 0
    except IndexingError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (PersistenceError, SQLAlchemyError, ValueError, OSError):
        print(
            "Incremental operation failed: check database, scan/report, scope and storage",
            file=sys.stderr,
        )
        return 1
    finally:
        if store is not None:
            store.client.close()
        if engine is not None:
            engine.dispose()


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


def run_prepare_policy_data(args: argparse.Namespace) -> int:
    evaluations = load_evaluations(args.evaluation)
    documents = load_corpora(args.corpus)
    manifest = load_manifest(args.splits)
    examples = build_policy_dataset(evaluations, documents, manifest)
    examples.extend(
        build_corpus_tool_examples(
            documents, manifest, limit=args.augment_training_from_corpus
        )
    )
    labeled_sources = {
        (example.repository, evidence.source_id)
        for example in examples
        for evidence in example.gold_evidence
    }
    from github_intelligence.post_training import DatasetSplit

    examples.extend(
        build_corpus_tool_examples(
            documents,
            manifest,
            limit=args.augment_test_from_corpus,
            split=DatasetSplit.TEST,
            excluded_sources=labeled_sources,
        )
    )
    write_policy_dataset(args.output, examples)
    summary = summarize_dataset(examples)
    print(f"Examples: {summary.examples}")
    for split, count in summary.split_counts.items():
        repositories = ", ".join(summary.repositories[split])
        print(f"{split.value}: {count} ({repositories})")
    print(f"Provenance label coverage: {summary.label_coverage:.4f}")
    print(f"Output: {args.output}")
    return 0


def run_train_policy(args: argparse.Namespace) -> int:
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be at least 1")
    if args.num_generations < 2:
        raise ValueError("num-generations must be at least 2")
    from github_intelligence.post_training.training import train_grpo, train_sft

    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "sft":
        train_sft(
            args.dataset,
            args.output,
            model_name=args.model,
            max_steps=args.max_steps or 75,
            seed=args.seed,
        )
    else:
        train_grpo(
            args.dataset,
            args.output,
            model_name=args.model,
            max_steps=args.max_steps or 150,
            seed=args.seed,
            num_generations=args.num_generations,
        )
    print(f"Saved {args.stage.upper()} adapter/checkpoint to {args.output}")
    return 0


def run_policy_evaluation(args: argparse.Namespace) -> int:
    from github_intelligence.post_training import DatasetSplit

    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be at least 1")
    report = evaluate_policy_checkpoint(
        args.dataset,
        args.model,
        args.output,
        split=DatasetSplit(args.split),
        max_new_tokens=args.max_new_tokens,
    )
    metrics = report.metrics
    print(f"Examples: {metrics.examples}")
    print(f"Tool-selection accuracy: {metrics.tool_selection_accuracy:.4f}")
    print(f"Structured action validity: {metrics.structured_action_validity:.4f}")
    print(f"Mean policy reward: {metrics.mean_policy_reward:.4f}")
    print(f"Malformed-output rate: {metrics.malformed_output_rate:.4f}")
    for group, group_metrics in report.metrics_by_group.items():
        print(
            f"{group}: n={group_metrics.examples}, "
            f"tool_accuracy={group_metrics.tool_selection_accuracy:.4f}, "
            f"validity={group_metrics.structured_action_validity:.4f}, "
            f"reward={group_metrics.mean_policy_reward:.4f}, "
            f"malformed={group_metrics.malformed_output_rate:.4f}"
        )
    print(f"Output: {args.output}")
    return 0


def run_application(args: argparse.Namespace, settings: Settings) -> int:
    import signal

    try:
        from github_intelligence.application.runtime import Runtime
        runtime = Runtime(settings)
    except (ImportError, ValueError):
        print(
            "Application startup failed: install the application extra and check configuration",
            file=sys.stderr,
        )
        return 1
    try:
        if args.command == "serve":
            import uvicorn

            from github_intelligence.application.api import create_app
            uvicorn.run(create_app(runtime=runtime), host=args.host, port=args.port,
                        access_log=False, log_level="warning", limit_concurrency=32)
        else:
            from github_intelligence.application.worker import Worker
            worker = Worker(runtime)
            previous = {}
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, lambda *_: worker.stop.set())
            try:
                if args.once:
                    worker.run_one()
                else:
                    worker.run()
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
        return 0
    finally:
        runtime.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings()
    configure_logging(args.log_level or settings.log_level)
    if args.command in {"serve", "worker"}:
        raise SystemExit(run_application(args, settings))
    if args.command in {"index-incremental", "retire-index"}:
        raise SystemExit(run_incremental(args, settings))
    if args.command == "import-metadata":
        raise SystemExit(run_import_metadata(args, settings))
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
    if args.command == "prepare-policy-data":
        raise SystemExit(run_prepare_policy_data(args))
    if args.command == "train-policy":
        raise SystemExit(run_train_policy(args))
    if args.command == "evaluate-policy":
        raise SystemExit(run_policy_evaluation(args))


if __name__ == "__main__":
    main()
