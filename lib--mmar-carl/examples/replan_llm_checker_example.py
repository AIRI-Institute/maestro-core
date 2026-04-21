#!/usr/bin/env python3
"""LLM RE-PLAN checker example with a multi-step trajectory and structured verdicts."""

from pathlib import Path
import re
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.utils import format_status  # noqa: E402
from mmar_carl import (  # noqa: E402
    ExecutionMode,
    Language,
    LLMClientBase,
    LLMStepConfig,
    LLMStepDescription,
    LLMReplanCheckerConfig,
    ReplanPolicy,
    ReasoningChain,
    ReasoningContext,
)


SAMPLE_CONTEXT = """
Task: Draft a product launch narrative for a risk-sensitive analytics rollout.
Required output:
- framing
- key risks
- final recommendation
Tone: practical and clear.
"""


class LLMCheckerTrajectoryClient(LLMClientBase):
    """Mock client for both step generation and LLM replan checker calls."""

    def __init__(self):
        self.critique_step_calls = 0
        self.did_replan_risk_step = False

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt, retries=1)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        _ = retries
        lowered = prompt.lower()

        if "chain-level replanning controller" in lowered:
            current_checker_step = None
            checker_step_match = re.search(r"- number:\s*(\d+)", lowered)
            if checker_step_match:
                current_checker_step = int(checker_step_match.group(1))

            if current_checker_step == 2 and not self.did_replan_risk_step:
                self.did_replan_risk_step = True
                return (
                    '{"action":"retry_current_step",'
                    '"reason":"Risk critique is generic and lacks explicit ownership",'
                    '"confidence":0.88,'
                    '"regeneration_hints":["Name concrete risk owners","Add one mitigation per risk"]}'
                )
            return '{"action":"continue","reason":"Critique is now actionable","confidence":0.93}'

        step_markers = re.findall(r"step\s+(\d+)\.", lowered)
        current_step = int(step_markers[-1]) if step_markers else None

        if current_step == 1:
            return "Framing complete: rollout impacts ingestion, dashboards, and alert thresholds."

        if current_step == 2:
            self.critique_step_calls += 1
            if self.critique_step_calls == 1:
                return "Risks exist but details are limited."
            return "Risks: drift (owner DataOps), false alerts (owner SRE), phased rollout mitigations included."

        if current_step == 3:
            return "Final recommendation: proceed with staged rollout and explicit owner-based mitigations."

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
            aim="Define rollout scope and operating constraints",
            reasoning_questions="Which systems and risks are in scope?",
            stage_action="Write scope framing",
            example_reasoning="Keep it concise and factual",
            llm_config=LLMStepConfig(execution_mode=ExecutionMode.FAST),
        ),
        LLMStepDescription(
            number=2,
            title="Risk Critique",
            aim="Produce concrete risks with owners and mitigations",
            reasoning_questions="Are risks specific and owned?",
            stage_action="Generate risk critique",
            example_reasoning="Each risk should map to one owner",
            dependencies=[1],
            checkpoint=True,
            checkpoint_name="risk_critique",
            llm_config=LLMStepConfig(execution_mode=ExecutionMode.FAST),
        ),
        LLMStepDescription(
            number=3,
            title="Final Synthesis",
            aim="Deliver final recommendation for launch readiness",
            reasoning_questions="Is the recommendation actionable and complete?",
            stage_action="Synthesize final output",
            example_reasoning="Tie recommendation to previous risks",
            dependencies=[2],
            llm_config=LLMStepConfig(execution_mode=ExecutionMode.SELF_CRITIC),
        ),
    ]

    policy = ReplanPolicy(
        enabled=True,
        checkers=[
            LLMReplanCheckerConfig(
                name="llm_quality_council",
                history_entries=8,
                error_entries=4,
            )
        ],
    )

    return ReasoningChain(steps=steps, max_workers=1, replan_policy=policy)


def print_chain_overview(chain: ReasoningChain) -> None:
    _section("Chain Overview")
    header = f"{'Step':<6} {'Mode':<12} {'Deps':<10} {'Checkpoint':<16} {'Title'}"
    print(header)
    print("-" * len(header))
    for step in chain.steps:
        llm_mode = step.llm_config.execution_mode.value if step.llm_config else "fast"
        deps = ",".join(str(dep) for dep in step.dependencies) if step.dependencies else "-"
        checkpoint = step.checkpoint_name if step.checkpoint else "-"
        print(f"{step.number:<6} {llm_mode:<12} {deps:<10} {checkpoint:<16} {step.title}")


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
        selected = event.aggregation.selected_checker or "-"
        print(
            f"step={event.step_number} action={event.final_action} selected_checker={selected} "
            f"votes={event.aggregation.trigger_count}/{event.aggregation.total_count} "
            f"feedback={event.feedback_passed}"
        )


def main() -> None:
    print("CARL RE-PLAN LLM Checker Trajectory Example")
    print("=" * 92)

    chain = build_chain()
    context = ReasoningContext(
        outer_context=SAMPLE_CONTEXT.strip(),
        api=LLMCheckerTrajectoryClient(),
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

    _section("RE-PLAN Summary")
    print(result.metadata.get("replan", {}))


if __name__ == "__main__":
    main()
