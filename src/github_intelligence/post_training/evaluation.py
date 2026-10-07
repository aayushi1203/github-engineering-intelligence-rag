from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from github_intelligence.post_training.dataset import load_policy_dataset
from github_intelligence.post_training.formatting import (
    POLICY_SYSTEM_PROMPT,
    completion_reward,
    parse_policy_action,
    policy_prompt,
)
from github_intelligence.post_training.models import DatasetSplit, PolicyExample
from github_intelligence.post_training.reward import action_schema_is_valid


class PolicyGenerator(Protocol):
    def generate(self, example: PolicyExample) -> str: ...


class PolicyDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    example_id: str
    benchmark_group: str = Field(pattern=r"^(core|corpus_derived)$")
    repository: str
    question: str
    acceptable_tools: list[str]
    completion: str
    parsed_tool: str | None
    tool_selection_correct: bool
    structured_action_valid: bool
    malformed: bool
    reward: float


class PolicyMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    examples: int = Field(ge=0)
    tool_selection_accuracy: float = Field(ge=0, le=1)
    structured_action_validity: float = Field(ge=0, le=1)
    mean_policy_reward: float = Field(ge=-0.5, le=1)
    malformed_output_rate: float = Field(ge=0, le=1)


class PolicyEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str
    dataset_path: str
    split: DatasetSplit
    metrics: PolicyMetrics
    metrics_by_group: dict[str, PolicyMetrics]
    diagnostics: list[PolicyDiagnostic]


def evaluate_policy(
    examples: list[PolicyExample], generator: PolicyGenerator, *, model: str, dataset_path: Path
) -> PolicyEvaluationReport:
    diagnostics: list[PolicyDiagnostic] = []
    for example in examples:
        completion = generator.generate(example)
        action = parse_policy_action(completion)
        malformed = action is None
        valid = action is not None and action_schema_is_valid(action, example.repository)
        correct = action is not None and action.tool in example.acceptable_tools
        diagnostics.append(
            PolicyDiagnostic(
                example_id=example.example_id,
                benchmark_group=(
                    "core" if example.example_id.startswith("policy-") else "corpus_derived"
                ),
                repository=example.repository,
                question=example.question,
                acceptable_tools=example.acceptable_tools,
                completion=completion,
                parsed_tool=action.tool if action else None,
                tool_selection_correct=correct,
                structured_action_valid=valid,
                malformed=malformed,
                reward=completion_reward(example, completion),
            )
        )
    metrics = _aggregate(diagnostics)
    groups = sorted({item.benchmark_group for item in diagnostics})
    return PolicyEvaluationReport(
        model=model,
        dataset_path=str(dataset_path),
        split=examples[0].split if examples else DatasetSplit.TEST,
        metrics=metrics,
        metrics_by_group={
            group: _aggregate([item for item in diagnostics if item.benchmark_group == group])
            for group in groups
        },
        diagnostics=diagnostics,
    )


def _aggregate(diagnostics: list[PolicyDiagnostic]) -> PolicyMetrics:
    count = len(diagnostics)
    denominator = count or 1
    return PolicyMetrics(
        examples=count,
        tool_selection_accuracy=sum(item.tool_selection_correct for item in diagnostics)
        / denominator,
        structured_action_validity=sum(item.structured_action_valid for item in diagnostics)
        / denominator,
        mean_policy_reward=sum(item.reward for item in diagnostics) / denominator,
        malformed_output_rate=sum(item.malformed for item in diagnostics) / denominator,
    )


def evaluate_policy_checkpoint(
    dataset_path: Path,
    model_name: str,
    output_path: Path,
    *,
    split: DatasetSplit = DatasetSplit.TEST,
    max_new_tokens: int = 192,
) -> PolicyEvaluationReport:
    examples = [item for item in load_policy_dataset(dataset_path) if item.split == split]
    if not examples:
        raise ValueError(f"Policy dataset contains no {split.value} examples")
    generator = TransformersPolicyGenerator(model_name, max_new_tokens=max_new_tokens)
    report = evaluate_policy(examples, generator, model=model_name, dataset_path=dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


class TransformersPolicyGenerator:
    def __init__(self, model_name: str, *, max_new_tokens: int = 192) -> None:
        from github_intelligence.post_training.training import load_inference_model

        self.model, self.tokenizer = load_inference_model(model_name)
        self.max_new_tokens = max_new_tokens

    def generate(self, example: PolicyExample) -> str:
        messages = [
            {"role": "system", "content": POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": policy_prompt(example)},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[-1]
        return str(self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True))


class CallablePolicyGenerator:
    """Small adapter useful for deterministic tests and offline diagnostics."""

    def __init__(self, function: Callable[[PolicyExample], str]) -> None:
        self.function = function

    def generate(self, example: PolicyExample) -> str:
        return self.function(example)
