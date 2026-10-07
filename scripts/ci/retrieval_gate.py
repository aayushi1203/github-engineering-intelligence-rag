"""Explicit gates for frozen inputs, report completeness and retrieval regression tolerances."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


class GateFailure(ValueError):
    pass


def input_hashes(config: dict[str, Any], root: Path) -> dict[str, dict[str, str]]:
    hashes = {}
    for key, specification in config["inputs"].items():
        observed = hashlib.sha256((root / specification["path"]).read_bytes()).hexdigest()
        hashes[key] = {**specification, "observed_sha256": observed}
    return hashes


def validate_inputs(hashes: dict[str, dict[str, str]]) -> None:
    for key, value in hashes.items():
        if value["observed_sha256"] != value["sha256"]:
            raise GateFailure(f"Frozen {key} hash mismatch")


def check_report(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    count = config["query_count"]
    if report.get("dataset_size") != count:
        raise GateFailure("Expected exactly 54 evaluation queries")
    modes = {"bm25", "vector", "hybrid", "hybrid_rerank"}
    metrics = report.get("metrics", {})
    if set(metrics) != modes:
        raise GateFailure("Report must include all four retrieval modes")
    for mode in modes:
        if metrics[mode].get("query_count") != count:
            raise GateFailure(f"Incomplete query metrics for {mode}")
        diagnostics = [d for d in report.get("diagnostics", []) if d.get("mode") == mode]
        if (
            len(diagnostics) != count
            or len({(d["repository"], d["query"]) for d in diagnostics}) != count
        ):
            raise GateFailure(f"Incomplete/duplicate query diagnostics for {mode}")
    failures = []
    comparisons: dict[str, Any] = {}
    for mode, gate in config["gates"].items():
        comparisons[mode] = {}
        for name in ("recall_at_10", "ndcg_at_10"):
            value, minimum = metrics[mode].get(name), gate["minimum_" + name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise GateFailure(f"Invalid {mode} {name}")
            comparisons[mode][name] = {"observed": value, "minimum": minimum}
            if value < minimum:
                failures.append(f"{mode} {name}: {value:.8f} < {minimum:.4f}")
    if failures:
        raise GateFailure("; ".join(failures))
    return {"passed": True, "comparisons": comparisons}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("ci/retrieval-benchmark.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_inputs(input_hashes(config, Path.cwd()))
    outcome = check_report(json.loads(args.report.read_text()), config)
    print(json.dumps(outcome, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
