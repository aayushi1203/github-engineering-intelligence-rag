# GitHub Engineering Intelligence Agent

The first five phases of a portfolio-grade engineering investigation system. It ingests public
GitHub repository activity and documentation into a typed document format, provides source-aware
BM25 and vector retrieval, combines both rankings with Reciprocal Rank Fusion (RRF), and can
rerank the fused evidence with a cross-encoder, then generate evidence-only answers with validated
GitHub citations. An offline source-labeled benchmark compares all four retrieval modes. Agents,
MCP, FastAPI, and UI remain out of scope.

## What it ingests

- Issues and issue comments (excluding pull requests returned by the issues endpoint)
- Pull requests, review comments, and conversation comments
- Commits
- Releases
- README and documentation files (`README*`, `docs/**`, and common Markdown/reStructuredText files)

Every item becomes a `GitHubDocument` with stable identity, source URL, timestamps, repository,
type, text, title, and source-specific metadata.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

A token is optional for public repositories. Add a fine-grained read-only token as
`GITHUB_TOKEN` to increase the API quota. Never commit `.env`.

## Usage

```bash
github-intel ingest encode/httpx --output output/httpx.jsonl
github-intel index output/httpx.jsonl
github-intel search "authentication timeout regression" --repo encode/httpx
github-intel search "authentication timeout regression" --repo encode/httpx --backend hybrid
github-intel search "authentication timeout regression" --repo encode/httpx \
  --backend hybrid_rerank --top-k 5
github-intel ask "What changed around this authentication regression?" --repo encode/httpx
```

Useful flags include `--no-docs`, `--no-comments`, and `--log-level DEBUG`. Output is JSON Lines,
which preserves the normalized Pydantic representation without selecting a future datastore.

The cache stores response bodies and ETags under `.cache/github`. Fresh entries avoid network
calls; stale entries are revalidated with `If-None-Match`. Pagination uses maximum-size page
boundaries, retries transient failures with exponential backoff, and waits when a rate-limit reset is
near (within a bounded interval). Set `GITHUB_MAX_PAGES` during development to cap large ingests.

### Retrieval backends

`index` builds both indexes by default. Use `--backend bm25` or `--backend vector` to build only
one. Search supports `bm25`, `vector`, `hybrid`, and `hybrid_rerank`; the default is
`hybrid_rerank`. Control the final result count with `--top-k`.

The vector backend uses embedded Qdrant, so no server, credentials, or Docker container is
required for local development. Its database path is configured with `QDRANT_PATH` (default
`.cache/qdrant`) and the collection with `QDRANT_COLLECTION`. `INDEX_DIR` controls the persisted
chunk file used to reconstruct BM25. All paths are relative to the working directory unless made
absolute.

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2` on CPU. The first vector
index/search downloads its weights from Hugging Face. Override `EMBEDDING_MODEL` and
`EMBEDDING_DEVICE` in `.env`; the configured model must use the same embedding dimension when
searching an existing collection, or the collection should be rebuilt.

Chunking uses headings and paragraphs for documentation and sentence/word windows only when a
logical unit exceeds `CHUNK_MAX_WORDS`. Issue and PR titles are repeated as chunk context;
comments preserve their parent identifiers and every chunk retains the original source URL and
metadata.

### Hybrid retrieval and reranking

BM25 is strongest when the query shares exact identifiers, error messages, API names, or other
terms with an engineering artifact. Vector retrieval can recover semantically related evidence
when the wording differs. Hybrid mode runs both independently, taking
`RETRIEVAL_CANDIDATES_PER_BACKEND` results from each (default 25).

RRF merges these lists without trying to calibrate incomparable lexical and cosine scores. For a
chunk at one-based rank `r`, a backend contributes `1 / (RRF_K + r)`; contributions are summed
across backends and chunks are deduplicated by stable chunk ID. `RRF_K` defaults to 60. The result
retains each backend's original rank and score for debugging.

In `hybrid_rerank` mode, the best `RERANK_CANDIDATE_POOL` fused chunks (default 40) are passed as
query–passage pairs to `cross-encoder/ms-marco-MiniLM-L-6-v2`. Unlike embedding similarity, the
cross-encoder jointly reads the query and candidate and supplies the final ordering. Configure it
with `RERANKER_MODEL` and `RERANKER_DEVICE`. Both embedding and reranker weights are downloaded on
first use; `bm25` and `hybrid` do not load the reranker.

### Grounded answer generation

`ask` always retrieves evidence with the existing `hybrid_rerank` mode, assigns the returned
chunks source IDs (`[S1]`, `[S2]`, ...), and sends an explicitly delimited evidence block to a
provider-neutral LLM interface. Set `OPENAI_API_KEY` in `.env`; the default provider uses the
OpenAI Responses API with strict structured output and configurable `OPENAI_MODEL` (default
`gpt-5.6-sol`). `GENERATION_TOP_K`, `GENERATION_MAX_CONTEXT_CHARS`, and
`GENERATION_MAX_OUTPUT_TOKENS` control generation inputs without changing retrieval settings.

Retrieved issues, PRs, comments, commits, releases, and documentation are treated as untrusted
data. The system prompt forbids following instructions inside them, and JSON serialization keeps
their content inside marked `retrieved_evidence` boundaries. The application—not the model—maps
citation IDs back to retrieved URLs. Inline references and the structured citation list must match
and may contain only retrieved IDs. Invalid output is repaired once by default and then rejected.

The response includes the answer, validated citations, cited sources, all retrieved sources, and
optional limitations. A deterministic evaluation helper reports citation validity, sentence-level
citation coverage, and lexical evidence support. The latter is a transparent proxy for
groundedness, not a model-judged faithfulness score and not a claimed benchmark result.

```text
                         ┌─ BM25 ───────┐
query + repository filter│              ├─ RRF + dedup ─ cross-encoder ─ final top-k
                         └─ Qdrant vector┘
```

## Retrieval evaluation

The JSONL benchmark format contains `repository`, `query`, `relevant_source_ids`, and optional
`notes`. Labels may be stable chunk IDs or originating GitHub/document IDs. Evaluation collapses
multiple retrieved chunks from the same source before scoring, preventing a long issue from
receiving repeated credit.

The starter [benchmark](data/eval/retrieval.jsonl) contains 20 known-item queries across
`fastapi/fastapi` and `pydantic/pydantic`: 10 per repository. Each label was constructed from an
issue inspected through GitHub's public API on 2026-08-20; its provenance URL is recorded in
`notes`. Queries conservatively paraphrase the corresponding issue title. This makes the dataset
useful for retrieval plumbing and regression testing, but it is not a substitute for independently
annotated, naturally occurring investigation questions. No parameters have been tuned on it. If
tuning begins, create a separate development JSONL and keep this file unchanged as the held-out
test set.

Metrics are macro-averaged across queries:

- **Recall@K:** fraction of a query's labeled sources present in the first K unique sources.
- **MRR:** reciprocal rank of the first relevant source, or zero when none is retrieved within the
  configured evaluation depth.
- **NDCG@10:** binary relevance gain discounted by `1/log2(rank+1)`, normalized against the ideal
  ranking containing up to 10 labeled sources.

Build the shared index from the checked-in benchmark corpus, then compare `bm25`, `vector`,
`hybrid`, and `hybrid_rerank`:

```bash
github-intel index data/corpus/benchmark.jsonl
github-intel evaluate-retrieval \
  --dataset data/eval/retrieval.jsonl \
  --top-k 10 \
  --output output/retrieval-evaluation.json
```

Use `--repo fastapi/fastapi` to evaluate a repository subset. The JSON report records model names,
RRF constant, candidate pools, per-mode aggregate metrics, retrieved source/chunk IDs, backend and
reranker scores, first relevant rank, and per-query success at K.

### Measured results

Measured on 2026-08-20 over all 20 queries. The complete machine-readable report, including 80
per-query/mode diagnostics, is at
[`output/retrieval-evaluation.json`](output/retrieval-evaluation.json). No parameters were tuned
on this benchmark.

| Backend | Recall@5 | Recall@10 | MRR | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 1.0000 | 1.0000 | 0.9417 | 0.9565 |
| Vector | 1.0000 | 1.0000 | 0.9167 | 0.9381 |
| Hybrid | 1.0000 | 1.0000 | 0.9417 | 0.9565 |
| Hybrid + reranker | 1.0000 | 1.0000 | 0.9417 | 0.9565 |

Per-repository results (10 queries each):

| Repository | Backend | Recall@5 | Recall@10 | MRR | NDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| `fastapi/fastapi` | BM25 | 1.0000 | 1.0000 | 0.9500 | 0.9631 |
| `fastapi/fastapi` | Vector | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `fastapi/fastapi` | Hybrid | 1.0000 | 1.0000 | 0.9500 | 0.9631 |
| `fastapi/fastapi` | Hybrid + reranker | 1.0000 | 1.0000 | 0.9500 | 0.9631 |
| `pydantic/pydantic` | BM25 | 1.0000 | 1.0000 | 0.9333 | 0.9500 |
| `pydantic/pydantic` | Vector | 1.0000 | 1.0000 | 0.8333 | 0.8762 |
| `pydantic/pydantic` | Hybrid | 1.0000 | 1.0000 | 0.9333 | 0.9500 |
| `pydantic/pydantic` | Hybrid + reranker | 1.0000 | 1.0000 | 0.9333 | 0.9500 |

The run completed with zero query failures or runtime errors. Reranking did not change the first
relevant rank for any query relative to hybrid, so there was no positive or negative reranking
delta and no “largest improvement” beyond zero. Hybrid + reranker was worse than the simpler
vector mode on one query: “Where is the frontend fallback 404 for client-side routes containing
dots such as john.doe discussed?” Vector ranked the labeled issue first; BM25, hybrid, and
hybrid + reranker ranked it second.

Exact benchmark configuration:

- embedding: `sentence-transformers/all-MiniLM-L6-v2`
- reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- RRF constant: 60
- candidates per backend: 25
- reranking pool: 40
- evaluation depth: 10
- dataset: 20 queries across 2 repositories (10 each)
- corpus: 632 documents / 928 chunks
- document mix: 32 issues, 200 pull requests, 200 commits, 200 releases
- index snapshot: one 100-item API page per source endpoint and repository, with comments and
  documentation disabled; eight labeled FastAPI issues absent from that page were fetched by
  issue number and normalized through the same Phase 1 client/normalizer
- label validation: all 20 GitHub source IDs matched exactly once before indexing

### Held-out benchmark v2

The harder [v2 dataset](data/eval/retrieval_v2.jsonl) is separate from v1 and contains 54 new
queries: 18 each for `fastapi/fastapi`, `pydantic/pydantic`, and the newly added `encode/httpx`.
It has 74 unique relevance labels, 18 multi-source questions, no relevance-label overlap with v1,
and provenance URLs for every label. Queries describe symptoms, configuration failures, release
investigations, behavioral changes, or related fix chains instead of copying artifact titles.

The dataset was frozen at SHA-256
`2bf0c1871b6c24c39f945cb40e2ffda954d07889eb343a1739c5818087a4057f` before indexing or
retrieval. The original v1 dataset remains unchanged at SHA-256
`6afb4f28c568547c57698562c2202db732f6e764952872dcbe1104f576a7befb`. No parameters were
tuned before or after the first v2 result reported here.

First held-out run on 2026-08-21:

| Backend | Recall@5 | Recall@10 | MRR | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.7608 | 0.8225 | 0.7068 | 0.7032 |
| Vector | 0.8519 | 0.8889 | 0.7722 | 0.7836 |
| Hybrid | 0.8225 | **0.9352** | **0.8013** | 0.8123 |
| Hybrid + reranker | **0.8688** | **0.9352** | 0.7972 | **0.8203** |

Per-repository results (18 queries each):

| Repository | Backend | Recall@5 | Recall@10 | MRR | NDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| `encode/httpx` | BM25 | 0.9074 | 0.9259 | 0.8056 | 0.7999 |
| `encode/httpx` | Vector | 0.9444 | 0.9444 | 0.8472 | 0.8573 |
| `encode/httpx` | Hybrid | 0.8981 | 1.0000 | 0.8690 | 0.8922 |
| `encode/httpx` | Hybrid + reranker | 0.8981 | 1.0000 | 0.8690 | 0.8922 |
| `fastapi/fastapi` | BM25 | 0.7361 | 0.8194 | 0.7083 | 0.7018 |
| `fastapi/fastapi` | Vector | 0.9444 | 0.9444 | 0.8194 | 0.8514 |
| `fastapi/fastapi` | Hybrid | 0.8472 | 1.0000 | 0.8032 | 0.8311 |
| `fastapi/fastapi` | Hybrid + reranker | 0.9028 | 1.0000 | 0.8079 | 0.8352 |
| `pydantic/pydantic` | BM25 | 0.6389 | 0.7222 | 0.6065 | 0.6080 |
| `pydantic/pydantic` | Vector | 0.6667 | 0.7778 | 0.6500 | 0.6421 |
| `pydantic/pydantic` | Hybrid | 0.7222 | 0.8056 | 0.7315 | 0.7137 |
| `pydantic/pydantic` | Hybrid + reranker | 0.8056 | 0.8056 | 0.7148 | 0.7334 |

Using per-query NDCG@10, vector beats BM25 on 20 queries, BM25 beats vector on 14, and hybrid
beats both on 4. Reranking helps 6 queries, hurts 5, and ties hybrid on 43. The largest reranking
gain is the Pydantic foreign-type URL comparison query (`0.6309` to `1.0000`); the largest loss is
the inherited TypedDict documentation query (`0.9197` to `0.5706`). The complete report and query
comparison lists are saved at
[`output/retrieval-evaluation-v2.json`](output/retrieval-evaluation-v2.json) and
[`output/retrieval-evaluation-v2-analysis.json`](output/retrieval-evaluation-v2-analysis.json).

The v2 run uses the original fixed settings: `all-MiniLM-L6-v2` embeddings,
`ms-marco-MiniLM-L-6-v2` reranking, RRF constant 60, 25 candidates per backend, a 40-candidate
reranking pool, and evaluation depth 10. Its corpus contains 892 normalized documents and 1,356
chunks. HTTPX has disabled public Issues, so unavailable `410` issue references were documented but
not labeled; all labels correspond to accessible ingested PRs, releases, or issues and matched the
corpus exactly once.

Reproduce the frozen run without changing settings:

```bash
github-intel index data/corpus/benchmark_v2.jsonl
HF_HUB_OFFLINE=1 github-intel evaluate-retrieval \
  --dataset data/eval/retrieval_v2.jsonl \
  --top-k 10 \
  --output output/retrieval-evaluation-v2.json
```

## Development

```bash
pytest
ruff check .
mypy src
```

## Architecture

```text
src/github_intelligence/
├── config.py                 # environment-backed settings
├── logging.py                # JSON structured logging
├── cli.py                    # ingestion command
├── models/document.py        # normalized contract
├── github/cache.py           # atomic disk cache + ETags
├── github/client.py          # REST, pagination, retries, rate limits
├── ingestion/
    ├── normalizers.py        # GitHub payload -> document adapters
│   └── service.py            # repository ingestion orchestration
├── evaluation/
│   ├── metrics.py            # Recall@K, MRR, NDCG@K
│   ├── models.py             # dataset and report contracts
│   └── evaluator.py          # comparison runner and diagnostics
├── generation/
│   ├── provider.py           # swappable LLM protocol + OpenAI Responses adapter
│   ├── context.py            # source IDs and untrusted evidence serialization
│   ├── prompts.py            # grounding and prompt-injection controls
│   ├── service.py            # retrieval, generation, citation validation/repair
│   ├── models.py             # structured answer and source contracts
│   └── evaluation.py         # deterministic answer-quality checks
└── retrieval/
    ├── chunking.py           # structure-aware source chunking
    ├── bm25.py               # persisted lexical index
    ├── vector.py             # embeddings + embedded Qdrant adapter
    ├── fusion.py             # reciprocal rank fusion
    ├── reranker.py           # lazy cross-encoder reranking
    └── pipeline.py           # bm25/vector/hybrid/hybrid_rerank modes
```
