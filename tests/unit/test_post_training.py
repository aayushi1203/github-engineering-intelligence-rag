from pathlib import Path

import pytest

from github_intelligence.evaluation.models import EvaluationExample
from github_intelligence.models import DocumentType, GitHubDocument
from github_intelligence.post_training import (
    CallablePolicyGenerator,
    DatasetSplit,
    EpisodeObservation,
    PolicyAction,
    SplitManifest,
    action_schema_is_valid,
    build_corpus_tool_examples,
    build_policy_dataset,
    completion_reward,
    evaluate_policy,
    evaluate_policy_action,
    load_policy_dataset,
    oracle_action,
    parse_policy_action,
    summarize_dataset,
    write_policy_dataset,
)
from github_intelligence.post_training.training import require_rl_stack


def document(identifier: str = "123") -> GitHubDocument:
    return GitHubDocument(
        text="The timeout regression was fixed.",
        document_type=DocumentType.PULL_REQUEST,
        repository="acme/train",
        title="Fix timeout regression",
        source_url=f"https://github.com/acme/train/pull/{identifier}",
        github_id=identifier,
    )


def manifest() -> SplitManifest:
    return SplitManifest(
        train=["acme/train"], development=["acme/dev"], test=["acme/test"]
    )


def example_dataset() -> list:
    evaluations = [
        EvaluationExample(
            repository="acme/train",
            query="Which change fixed the timeout regression?",
            relevant_source_ids=["123"],
        )
    ]
    return build_policy_dataset(evaluations, [document()], manifest())


def test_policy_dataset_preserves_provenance_and_repository_split(tmp_path: Path) -> None:
    examples = example_dataset()
    assert examples[0].split == DatasetSplit.TRAIN
    assert examples[0].acceptable_tools == ["search_pull_requests"]
    assert str(examples[0].gold_evidence[0].source_url).endswith("/pull/123")
    output = tmp_path / "policy.jsonl"
    write_policy_dataset(output, examples)
    assert load_policy_dataset(output) == examples
    summary = summarize_dataset(examples)
    assert summary.label_coverage == 1.0
    assert summary.split_counts[DatasetSplit.TRAIN] == 1


def test_policy_dataset_rejects_missing_labels() -> None:
    evaluation = EvaluationExample(
        repository="acme/train", query="Missing?", relevant_source_ids=["missing"]
    )
    with pytest.raises(ValueError, match="Missing relevance label"):
        build_policy_dataset([evaluation], [document()], manifest())


def test_split_manifest_requires_disjoint_repositories() -> None:
    duplicate = SplitManifest(
        train=["acme/train"], development=["acme/train"], test=["acme/test"]
    )
    with pytest.raises(ValueError, match="disjoint"):
        _ = duplicate.repositories


def test_action_schema_validation() -> None:
    valid = PolicyAction(
        action="tool_call",
        tool="search_pull_requests",
        arguments={"repository": "acme/train", "query": "timeout"},
        reason_code="locate_fix",
    )
    assert action_schema_is_valid(valid, "acme/train")
    wrong_repository = valid.model_copy(
        update={"arguments": {"repository": "acme/other", "query": "timeout"}}
    )
    assert not action_schema_is_valid(wrong_repository, "acme/train")


def test_programmatic_reward_is_deterministic() -> None:
    example = example_dataset()[0]
    action = PolicyAction(
        action="tool_call",
        tool="search_pull_requests",
        arguments={"repository": "acme/train", "query": "timeout regression"},
        reason_code="locate_fix",
    )
    observation = EpisodeObservation(
        acquired_source_ids=["123"], tool_succeeded=True, call_count=1, call_budget=5
    )
    first = evaluate_policy_action(example, action, observation)
    second = evaluate_policy_action(example, action, observation)
    assert first == second
    assert first.total == pytest.approx(0.89)


def test_reward_penalizes_invalid_identifier_and_duplicate_call() -> None:
    example = example_dataset()[0]
    action = PolicyAction(
        action="tool_call",
        tool="get_pull_request",
        arguments={"repository": "acme/train", "number": 999},
        reason_code="inspect_fix",
    )
    reward = evaluate_policy_action(
        example,
        action,
        EpisodeObservation(
            tool_succeeded=False,
            duplicate_call=True,
            invalid_identifier=True,
            call_count=5,
            call_budget=5,
        ),
    )
    assert reward.total < 0


def test_policy_action_rejects_malformed_shape() -> None:
    with pytest.raises(ValueError, match="require a tool"):
        PolicyAction(action="tool_call", reason_code="bad")
    with pytest.raises(ValueError, match="cannot contain"):
        PolicyAction(
            action="finish",
            tool="search_issues",
            arguments={},
            reason_code="done",
        )


def test_corpus_augmentation_is_training_only_and_provenance_backed() -> None:
    examples = build_corpus_tool_examples([document()], manifest(), limit=10)
    assert len(examples) == 1
    assert examples[0].split == DatasetSplit.TRAIN
    assert examples[0].gold_evidence[0].source_id == "123"
    assert str(examples[0].gold_evidence[0].source_url).endswith("/pull/123")


def test_corpus_augmentation_respects_requested_split_and_exclusions() -> None:
    held_out = document().model_copy(update={"repository": "acme/test"})
    examples = build_corpus_tool_examples(
        [held_out], manifest(), limit=10, split=DatasetSplit.TEST
    )
    assert examples[0].split == DatasetSplit.TEST
    excluded = build_corpus_tool_examples(
        [held_out],
        manifest(),
        limit=10,
        split=DatasetSplit.TEST,
        excluded_sources={("acme/test", "123")},
    )
    assert excluded == []


def test_corpus_augmentation_balances_available_tool_classes() -> None:
    issue = document("1").model_copy(
        update={
            "document_type": DocumentType.ISSUE,
            "source_url": "https://github.com/acme/train/issues/1",
        }
    )
    pull = document("2")
    documentation = document("3").model_copy(
        update={
            "document_type": DocumentType.DOCUMENTATION,
            "source_url": "https://github.com/acme/train/blob/main/README.md",
        }
    )
    extras = [document(str(number)) for number in range(4, 10)]
    examples = build_corpus_tool_examples(
        [*extras, issue, pull, documentation], manifest(), limit=3
    )
    assert {item.acceptable_tools[0] for item in examples} == {
        "search_issues",
        "search_pull_requests",
        "search_repository_documents",
    }


def test_oracle_completion_and_reward_are_machine_verifiable() -> None:
    example = example_dataset()[0]
    completion = oracle_action(example).model_dump_json()
    assert parse_policy_action(completion) == oracle_action(example)
    assert completion_reward(example, completion) == 1.0
    assert completion_reward(example, "not json") == -0.5


def test_missing_rl_stack_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", fail_import)
    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[rl\]'"):
        require_rl_stack()


def test_policy_evaluation_uses_frozen_deterministic_formulas(tmp_path: Path) -> None:
    examples = example_dataset()
    completion = oracle_action(examples[0]).model_dump_json()
    report = evaluate_policy(
        examples,
        CallablePolicyGenerator(lambda _: completion),
        model="fake",
        dataset_path=tmp_path / "policy.jsonl",
    )
    assert report.metrics.examples == 1
    assert report.metrics.tool_selection_accuracy == 1.0
    assert report.metrics.structured_action_validity == 1.0
    assert report.metrics.mean_policy_reward == 1.0
    assert report.metrics.malformed_output_rate == 0.0
    assert report.metrics_by_group["core"].examples == 1


def test_policy_evaluation_counts_malformed_outputs(tmp_path: Path) -> None:
    report = evaluate_policy(
        example_dataset(),
        CallablePolicyGenerator(lambda _: "malformed"),
        model="fake",
        dataset_path=tmp_path / "policy.jsonl",
    )
    assert report.metrics.tool_selection_accuracy == 0.0
    assert report.metrics.structured_action_validity == 0.0
    assert report.metrics.mean_policy_reward == -0.5
    assert report.metrics.malformed_output_rate == 1.0
