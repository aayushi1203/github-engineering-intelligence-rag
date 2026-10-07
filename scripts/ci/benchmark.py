"""Isolated legacy full-snapshot benchmark with pinned, verified weights and explicit gates."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from scripts.ci.retrieval_gate import check_report, input_hashes, validate_inputs


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare_models(config: dict[str, Any], *, offline: bool) -> dict[str, str]:
    import hashlib

    from huggingface_hub import hf_hub_download

    paths = {}
    for role, model in config["models"].items():
        root = None
        for filename, expected in model["files"].items():
            path = Path(
                hf_hub_download(
                    model["repository"],
                    filename,
                    revision=model["revision"],
                    local_files_only=offline,
                    token=False,
                )
            )
            candidate = path
            for _ in Path(filename).parts:
                candidate = candidate.parent
            if candidate.name != model["revision"] or (root is not None and root != candidate):
                raise ValueError("Model snapshot does not match its pinned revision")
            root = candidate
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Pinned {role} file checksum mismatch: {filename}")
        if root is None:
            raise ValueError("Pinned model has no required files")
        paths[role] = str(root)
    return paths


def run(output: Path, *, config_path: Path, offline: bool) -> int:
    root = Path.cwd()
    output.mkdir(parents=True, exist_ok=True)
    try:
        config = json.loads(config_path.read_text())
        write_json(
            output / "configuration.json",
            {
                "benchmark": config,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "sentence-transformers",
                        "transformers",
                        "qdrant-client",
                        "numpy",
                    )
                },
            },
        )
        hashes = input_hashes(config, root)
        write_json(output / "input-hashes.json", hashes)
        validate_inputs(hashes)
        models = prepare_models(config, offline=offline)
        write_json(output / "models.json", {"pins": config["models"], "resolved_paths": models})
        # Download only public pinned files, then model inference is entirely offline.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from github_intelligence import cli
        from github_intelligence.config import Settings
        from github_intelligence.evaluation import RetrievalEvaluator, load_evaluation_dataset
        from github_intelligence.retrieval import RetrievalMode
        from github_intelligence.retrieval.vector import QdrantVectorRetriever, QdrantVectorStore

        retrieval = dict(config["retrieval"])
        depth = retrieval.pop("retrieval_depth")
        examples = load_evaluation_dataset(root / config["inputs"]["evaluation"]["path"])
        if len(examples) != config["query_count"]:
            raise ValueError("Frozen benchmark must contain exactly 54 queries")
        with tempfile.TemporaryDirectory(prefix="github-intel-ci-benchmark-") as temporary:
            settings_options: dict[str, Any] = dict(
                _env_file=None,
                database_url=None,
                github_token=None,
                openai_api_key=None,
                embedding_model=models["embedding"],
                reranker_model=models["reranker"],
                embedding_device="cpu",
                reranker_device="cpu",
                **retrieval,
                index_dir=Path(temporary) / "index",
                qdrant_path=Path(temporary) / "qdrant",
                qdrant_collection="github_chunks",
            )
            settings = Settings(**settings_options)
            env = os.environ.copy()
            env.update(
                {
                    key.upper(): str(value)
                    for key, value in {
                        **retrieval,
                        "index_dir": settings.index_dir,
                        "qdrant_path": settings.qdrant_path,
                        "qdrant_collection": settings.qdrant_collection,
                        "embedding_model": models["embedding"],
                        "reranker_model": models["reranker"],
                        "embedding_device": "cpu",
                        "reranker_device": "cpu",
                        "database_url": "",
                        "github_token": "",
                        "openai_api_key": "",
                    }.items()
                }
            )
            with (output / "benchmark.log").open("w") as log:
                # Existing CLI; process exit releases embedded Qdrant before evaluation.
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "github_intelligence.cli",
                        "index",
                        str(root / config["inputs"]["corpus"]["path"]),
                        "--backend",
                        "both",
                    ],
                    cwd=temporary,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=600,
                )
            pipeline = cli.build_retrieval_pipeline(settings, set(RetrievalMode))
            try:
                report = RetrievalEvaluator(pipeline, settings).evaluate(
                    examples,
                    dataset_path=root / config["inputs"]["evaluation"]["path"],
                    modes=tuple(RetrievalMode),
                    retrieval_depth=depth,
                )
                # Pydantic serializes existing non-finite diagnostic scores as null;
                # aggregate metrics are still required to be finite by the gate.
                data = json.loads(report.model_dump_json())
                write_json(output / "report.json", data)
                with (output / "per-query.jsonl").open("w") as handle:
                    for diagnostic in data["diagnostics"]:
                        handle.write(json.dumps(diagnostic, allow_nan=False) + "\n")
                write_json(output / "quality-gate.json", check_report(data, config))
                print(json.dumps(data["metrics"], indent=2))
            finally:
                if isinstance(pipeline.vector, QdrantVectorRetriever) and isinstance(
                    pipeline.vector.store, QdrantVectorStore
                ):
                    pipeline.vector.store.client.close()
        return 0
    except Exception as error:
        write_json(
            output / "quality-gate.json",
            {"passed": False, "error_type": type(error).__name__, "detail": str(error)},
        )
        print(f"Benchmark failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/ci-retrieval"))
    parser.add_argument("--config", type=Path, default=Path("ci/retrieval-benchmark.json"))
    parser.add_argument(
        "--offline", action="store_true", help="Use only already cached pinned files"
    )
    args = parser.parse_args()
    return run(args.output.resolve(), config_path=args.config, offline=args.offline)


if __name__ == "__main__":
    raise SystemExit(main())
