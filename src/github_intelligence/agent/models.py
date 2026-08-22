from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from github_intelligence.generation import GroundedAnswer


class ToolAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    arguments: dict[str, Any]
    purpose: str


class InvestigationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[str] = Field(min_length=1)
    actions: list[ToolAction] = Field(default_factory=list)


class InvestigationTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int = Field(ge=1)
    action: str
    detail: str


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    arguments: dict[str, Any]
    succeeded: bool
    evidence_count: int = Field(ge=0)
    error: str | None = None


class InvestigationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repository: str
    question: str
    plan: InvestigationPlan
    answer: GroundedAnswer
    investigation_steps: list[InvestigationTrace]
    tool_calls: list[ToolCallRecord]
    limitations: list[str] = Field(default_factory=list)
