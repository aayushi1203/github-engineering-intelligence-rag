"""Gate behavior: exact input contracts, boundary failures, and preserved formatting debt."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.ci.check_format import checked_files
from scripts.ci.retrieval_gate import GateFailure, check_report, input_hashes, validate_inputs

ROOT = Path(__file__).resolve().parents[2]


def config():
    return json.loads((ROOT / "ci/retrieval-benchmark.json").read_text())


def report():
    modes = ["bm25", "vector", "hybrid", "hybrid_rerank"]
    return {
        "dataset_size": 54,
        "metrics": {
            mode: {"query_count": 54, "recall_at_10": 0.9352, "ndcg_at_10": 0.8203}
            for mode in modes
        },
        "diagnostics": [
            {"mode": mode, "repository": "acme/repo", "query": str(n)}
            for mode in modes
            for n in range(54)
        ],
    }


def test_frozen_inputs_and_threshold_contract():
    specification = config()
    validate_inputs(input_hashes(specification, ROOT))
    assert specification["query_count"] == 54
    assert specification["gates"]["hybrid"]["minimum_recall_at_10"] == 0.9252
    assert specification["gates"]["hybrid"]["minimum_ndcg_at_10"] == 0.7923
    assert specification["gates"]["hybrid_rerank"]["minimum_ndcg_at_10"] == 0.8003
    assert check_report(report(), specification)["passed"]


@pytest.mark.parametrize("mode", ["hybrid", "hybrid_rerank"])
@pytest.mark.parametrize("metric", ["recall_at_10", "ndcg_at_10"])
def test_exact_threshold_passes_and_below_it_fails(mode, metric):
    example = report()
    minimum = config()["gates"][mode]["minimum_" + metric]
    example["metrics"][mode][metric] = minimum
    assert check_report(example, config())["passed"]
    example["metrics"][mode][metric] = minimum - 0.000001
    with pytest.raises(GateFailure):
        check_report(example, config())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.99", -1])
def test_invalid_metric_fails(value):
    example = report()
    example["metrics"]["hybrid_rerank"]["ndcg_at_10"] = value
    with pytest.raises(GateFailure):
        check_report(example, config())


@pytest.mark.parametrize("damage", ["count", "mode", "diagnostic", "duplicate", "mode_count"])
def test_incomplete_reports_fail(damage):
    example = copy.deepcopy(report())
    if damage == "count":
        example["dataset_size"] = 53
    elif damage == "mode":
        example["metrics"].pop("vector")
    elif damage == "diagnostic":
        example["diagnostics"].pop()
    elif damage == "duplicate":
        example["diagnostics"][0] = example["diagnostics"][1]
    else:
        example["metrics"]["vector"]["query_count"] = 53
    with pytest.raises(GateFailure):
        check_report(example, config())


def test_input_tampering_fails_even_with_a_previous_report(tmp_path):
    source = tmp_path / "frozen.jsonl"
    source.write_text("original")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    specification = {"inputs": {"corpus": {"path": source.name, "sha256": expected}}}
    source.write_text("changed")
    with pytest.raises(GateFailure, match="hash"):
        validate_inputs(input_hashes(specification, tmp_path))


def test_only_unchanged_formatting_debt_is_exempt(tmp_path):
    directory = tmp_path / "src"
    directory.mkdir()
    legacy = directory / "legacy.py"
    legacy.write_text("x=1\n")
    debt = {"src/legacy.py": hashlib.sha256(legacy.read_bytes()).hexdigest()}
    assert checked_files(tmp_path, debt) == []
    legacy.write_text("x=2\n")
    new = directory / "new.py"
    new.write_text("y=3\n")
    assert checked_files(tmp_path, debt) == ["src/legacy.py", "src/new.py"]
