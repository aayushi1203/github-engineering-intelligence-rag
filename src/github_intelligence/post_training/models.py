from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from github_intelligence.models import DocumentType


class DatasetSplit(str, Enum):  # noqa: UP042
    TRAIN = "train"
    DEVELOPMENT = "development"
    TEST = "test"


class PolicyAction(BaseModel):
    """One public, bounded action emitted by a trainable investigation policy."""

    model_config = ConfigDict(extra="forbid")
    action: str = Field(pattern=r"^(tool_call|finish)$")
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason_code: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_action_shape(self) -> "PolicyAction":
        if self.action == "tool_call" and not self.tool:
            raise ValueError("tool_call actions require a tool")
        if self.action == "finish" and (self.tool is not None or self.arguments):
            raise ValueError("finish actions cannot contain a tool or arguments")
        return self


class GoldEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    source_url: HttpUrl
    source_type: DocumentType
    title: str


class PolicyExample(BaseModel):
    """A provenance-preserving policy-training or evaluation example."""

    model_config = ConfigDict(extra="forbid")
    example_id: str
    repository: str = Field(pattern=r"^[^/]+/[^/]+$")
    question: str = Field(min_length=1)
    split: DatasetSplit
    gold_evidence: list[GoldEvidence] = Field(min_length=1)
    acceptable_tools: list[str] = Field(min_length=1)
    notes: str | None = None


class RewardBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total: float
    evidence_success: float
    schema_validity: float
    tool_success: float
    duplicate_penalty: float
    call_cost_penalty: float
    invalid_identifier_penalty: float


class EpisodeObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    acquired_source_ids: list[str] = Field(default_factory=list)
    tool_succeeded: bool = False
    duplicate_call: bool = False
    call_count: int = Field(default=1, ge=0)
    call_budget: int = Field(default=5, ge=1)
    invalid_identifier: bool = False


class FeasibilitySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    examples: int = Field(ge=0)
    repositories: dict[DatasetSplit, list[str]]
    split_counts: dict[DatasetSplit, int]
    label_coverage: float = Field(ge=0, le=1)
    reward_agreement: float | None = Field(default=None, ge=0, le=1)

