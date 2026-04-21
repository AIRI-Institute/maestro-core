#!/usr/bin/env python3
"""Deterministic RE-PLAN example with a multi-step LLM trajectory."""

from pathlib import Path
import re
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.utils import format_status  # noqa: E402
from mmar_carl import (  # noqa: E402
    Language,
    LLMClientBase,
    LLMStepConfig,
    LLMStepDescription,
    ReplanAction,
    ReplanPolicy,
    ReasoningChain,
    ReasoningContext,
    RuleBasedReplanCheckerConfig,
)


SAMPLE_CONTEXT = """
Task: Prepare a concise launch-risk memo for a new analytics feature.
Required sections:
- Scope summary
- Key risks
- Mitigation actions
Tone: practical and concise.
"""


class DeterministicTrajectoryClient(LLMClientBase):
    """Mock client that returns deterministic outputs by step intent."""

    def __init__(self):
        self.risk_calls = 0

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt, retries=1)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        _ = retries
        lowered = prompt.lower()
        step_markers = re.findall(r"step\s+(\d+)\.", lowered)
        current_step = int(step_markers[-1]) if step_markers else None

        if current_step == 1:
            return "Scope framing complete: analytics rollout touches ingestion, UI, and alert rules."

        if current_step == 2:
            self.risk_calls += 1
            if self.risk_calls == 1:
                return "NEEDS_REPLAN: risk audit is too generic and lacks concrete mitigation ownership."
            return "Risk audit complete: data drift, alert fatigue, and rollout blast radius with mitigations."

        if current_step == 3:
            return "Launch-risk memo finalized with explicit mitigation owners and rollout guardrails."

        return "fallback"


def _section(title: str) -> None:
    line = "=" * 92
    print(f"\n{line}\n{title}\n{line}")


def _truncate(text: str, width: int = 96) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= width:
        return cleaned
    return cleaned[: width - 3] + "..."


def build_chain() -> ReasoningChain:
    steps = [
        LLMStepDescription(
            number=1,
            title="Initial Framing",
            aim="Frame scope and boundaries of the launch-risk memo",
            reasoning_questions="What systems and responsibilities are in scope?",
            stage_action="Build concise scope framing",
            example_reasoning="List only systems relevant to launch risk",
            llm_config=LLMStepConfig(),
        ),
        LLMStepDescription(
            number=2,
            title="Risk Audit",
            aim="Identify concrete launch risks and mitigation owners",
            reasoning_questions="Which risks are most likely and who owns mitigation?",
            stage_action="Produce actionable risk table",
            example_reasoning="Each risk must include at least one mitigation owner",
            dependencies=[1],
            checkpoint=True,
            checkpoint_name="risk_phase",
            llm_config=LLMStepConfig(),
        ),
        LLMStepDescription(
            number=3,
            title="Final Synthesis",
            aim="Produce final memo with crisp recommendations",
            reasoning_questions="Does the memo include scope, risks, and mitigations?",
            stage_action="Synthesize final output",
            example_reasoning="Be concise but complete",
            dependencies=[2],
            llm_config=LLMStepConfig(),
        ),
    ]

    policy = ReplanPolicy(
        enabled=True,
        checkers=[
            RuleBasedReplanCheckerConfig(
                name="risk_quality_guard",
                result_substrings=["NEEDS_REPLAN"],
                action_on_match=ReplanAction.RETRY_CURRENT_STEP,
                feedback_on_match=["Add explicit mitigation ownership and concrete risks."],
            )
        ],
    )

    return ReasoningChain(steps=steps, max_workers=1, replan_policy=policy)


def print_chain_overview(chain: ReasoningChain) -> None:
    _section("Chain Overview")
    header = f"{'Step':<6} {'Deps':<10} {'Checkpoint':<14} {'Title'}"
    print(header)
    print("-" * len(header))
    for step in chain.steps:
        deps = ",".join(str(dep) for dep in step.dependencies) if step.dependencies else "-"
        checkpoint = step.checkpoint_name if step.checkpoint else "-"
        print(f"{step.number:<6} {deps:<10} {checkpoint:<14} {step.title}")


def print_step_results(result: Any) -> None:
    _section("Step Results")
    header = f"{'Step':<6} {'Status':<10} {'Time':<8} {'Output'}"
    print(header)
    print("-" * len(header))
    for step_result in sorted(result.step_results, key=lambda item: item.step_number):
        status = format_status(step_result.success)
        duration = f"{step_result.execution_time:.2f}s" if step_result.execution_time is not None else "n/a"
        text = step_result.result if step_result.success else (step_result.error_message or "")
        print(f"{step_result.step_number:<6} {status:<10} {duration:<8} {_truncate(text)}")


def print_replan_events(result: Any) -> None:
    _section("RE-PLAN Events")
    if not result.replan_events:
        print("No RE-PLAN events recorded.")
        return

    for event in result.replan_events:
        print(
            f"step={event.step_number} action={event.final_action} "
            f"trigger_votes={event.aggregation.trigger_count}/{event.aggregation.total_count} "
            f"target={event.rollback_target}"
        )


def main() -> None:
    print("CARL RE-PLAN Deterministic Trajectory Example")
    print("=" * 92)

    chain = build_chain()
    context = ReasoningContext(
        outer_context=SAMPLE_CONTEXT.strip(),
        api=DeterministicTrajectoryClient(),
        model="unused",
        language=Language.ENGLISH,
    )

    print_chain_overview(chain)
    _section("Execution")
    result = chain.execute(context)

    print(f"Success: {format_status(result.success)}")
    if result.total_execution_time is not None:
        print(f"Total time: {result.total_execution_time:.2f}s")
    print_step_results(result)
    print_replan_events(result)

    _section("Final Output")
    print(result.get_final_output())


if __name__ == "__main__":
    main()
