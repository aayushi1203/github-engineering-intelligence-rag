from typing import Protocol

from github_intelligence.agent.models import (
    InvestigationResult,
    InvestigationTrace,
    ToolCallRecord,
)
from github_intelligence.agent.planner import InvestigationPlanner
from github_intelligence.generation import GroundedAnswerService
from github_intelligence.mcp import ToolEvidence, ToolEvidenceResult
from github_intelligence.retrieval import RetrievalMode, RetrievalPipeline


class ToolExecutor(Protocol):
    async def call(self, name: str, arguments: dict[str, object]) -> ToolEvidenceResult: ...


class InvestigationAgent:
    def __init__(
        self,
        pipeline: RetrievalPipeline,
        answer_service: GroundedAnswerService,
        tools: ToolExecutor,
        *,
        planner: InvestigationPlanner | None = None,
        max_steps: int = 6,
        max_mcp_calls: int = 4,
        max_sources: int = 12,
        search_limit: int = 5,
    ) -> None:
        self.pipeline = pipeline
        self.answer_service = answer_service
        self.tools = tools
        self.planner = planner or InvestigationPlanner()
        self.max_steps = max_steps
        self.max_mcp_calls = max_mcp_calls
        self.max_sources = max_sources
        self.search_limit = search_limit

    async def investigate(self, question: str, *, repository: str) -> InvestigationResult:
        plan = self.planner.plan(question, repository, search_limit=self.search_limit)
        trace: list[InvestigationTrace] = []
        calls: list[ToolCallRecord] = []
        limitations: list[str] = []
        initial = self.pipeline.search(
            question,
            mode=RetrievalMode.HYBRID_RERANK,
            top_k=min(self.search_limit, self.max_sources),
            repository=repository,
        )
        evidence: dict[str, ToolEvidence] = {}
        self._add_evidence(evidence, [ToolEvidence.from_retrieval(item) for item in initial])
        trace.append(
            InvestigationTrace(
                step=1,
                action="retrieval",
                detail=f"Retrieved {len(evidence)} hybrid-reranked sources from {repository}",
            )
        )

        pending = list(plan.actions)
        while (
            pending
            and len(trace) < self.max_steps - 1
            and len(calls) < self.max_mcp_calls
            and len(evidence) < self.max_sources
        ):
            action = pending.pop(0)
            try:
                result = await self.tools.call(action.tool_name, action.arguments)
                added = self._add_evidence(evidence, result.items)
                calls.append(
                    ToolCallRecord(
                        tool_name=action.tool_name,
                        arguments=action.arguments,
                        succeeded=True,
                        evidence_count=added,
                    )
                )
                trace.append(
                    InvestigationTrace(
                        step=len(trace) + 1,
                        action=action.tool_name,
                        detail=f"{action.purpose}; added {added} deduplicated sources",
                    )
                )
            except Exception as error:
                calls.append(
                    ToolCallRecord(
                        tool_name=action.tool_name,
                        arguments=action.arguments,
                        succeeded=False,
                        evidence_count=0,
                        error=str(error),
                    )
                )
                trace.append(
                    InvestigationTrace(
                        step=len(trace) + 1,
                        action=action.tool_name,
                        detail=f"Tool failed safely: {type(error).__name__}",
                    )
                )
                limitations.append(f"{action.tool_name} failed: {error}")
            if len(evidence) >= self.max_sources:
                limitations.append(f"Evidence was capped at {self.max_sources} sources.")
                break

        if pending:
            if len(calls) >= self.max_mcp_calls:
                limitations.append(f"MCP tool-call limit ({self.max_mcp_calls}) was reached.")
            if len(trace) >= self.max_steps - 1:
                limitations.append(f"Investigation step limit ({self.max_steps}) was reached.")

        if (
            calls
            and len(trace) < self.max_steps - 1
            and len(calls) < self.max_mcp_calls
            and len(evidence) < self.max_sources
        ):
            terms = " ".join(item.title for item in list(evidence.values())[-3:])
            follow_up_arguments = {
                "repository": repository,
                "query": f"{question} {terms}",
                "limit": self.search_limit,
            }
            try:
                result = await self.tools.call("search_repository_documents", follow_up_arguments)
                added = self._add_evidence(evidence, result.items)
                calls.append(
                    ToolCallRecord(
                        tool_name="search_repository_documents",
                        arguments=follow_up_arguments,
                        succeeded=True,
                        evidence_count=added,
                    )
                )
                trace.append(
                    InvestigationTrace(
                        step=len(trace) + 1,
                        action="search_repository_documents",
                        detail=f"Follow-up retrieval added {added} deduplicated sources",
                    )
                )
            except Exception as error:
                calls.append(
                    ToolCallRecord(
                        tool_name="search_repository_documents",
                        arguments={"repository": repository},
                        succeeded=False,
                        evidence_count=0,
                        error=str(error),
                    )
                )
                limitations.append(f"Follow-up retrieval failed: {error}")

        selected = list(evidence.values())[: self.max_sources]
        answer = self.answer_service.answer_from_results(
            question,
            repository=repository,
            results=[item.as_retrieval_result() for item in selected],
        )
        if len(trace) < self.max_steps:
            trace.append(
                InvestigationTrace(
                    step=len(trace) + 1,
                    action="synthesis",
                    detail=f"Generated answer from {len(selected)} evidence sources",
                )
            )
        else:
            limitations.append(f"Investigation step limit ({self.max_steps}) was reached.")
        combined_limitations = list(dict.fromkeys(limitations + (answer.limitations or [])))
        if combined_limitations != (answer.limitations or []):
            answer = answer.model_copy(update={"limitations": combined_limitations or None})
        return InvestigationResult(
            repository=repository,
            question=question,
            plan=plan,
            answer=answer,
            investigation_steps=trace,
            tool_calls=calls,
            limitations=combined_limitations,
        )

    def _add_evidence(self, evidence: dict[str, ToolEvidence], items: list[ToolEvidence]) -> int:
        before = len(evidence)
        existing_urls = {str(item.source_url) for item in evidence.values()}
        for item in items:
            if len(evidence) >= self.max_sources:
                break
            url = str(item.source_url)
            if item.repository and url not in existing_urls:
                evidence[item.evidence_id] = item
                existing_urls.add(url)
        return len(evidence) - before
