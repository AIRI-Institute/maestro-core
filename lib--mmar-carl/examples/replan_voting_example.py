#!/usr/bin/env python3
"""Multi-checker voting (council) RE-PLAN example with a multi-step LLM trajectory."""

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
    ReplanAction,
    ReplanAggregationConfig,
    ReplanAggregationStrategy,
    ReplanCheckerBase,
    ReplanCheckerInput,
    ReplanPolicy,
    ReplanVerdict,
    ReasoningChain,
    ReasoningContext,
    RegisteredReplanCheckerConfig,
)


SAMPLE_CONTEXT = """
Task: Build a trajectory for launch readiness recommendations.
Required sections:
- framing
- risk council validation
- final recommendation
"""


class CouncilTrajectoryClient(LLMClientBase):
    """Mock client that returns a risky middle-step output once."""

    def __init__(self):
        self.council_calls = 0

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt, retries=1)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        _ = retries
        lowered = prompt.lower()
        step_markers = re.findall(r"step\s+(\d+)\.", lowered)
        current_step = int(step_markers[-1]) if step_markers else None

        if current_step == 1:
            return "Framing complete: scope includes ingestion, dashboards, and alert governance."

        if current_step == 2:
            self.council_calls += 1
            if self.council_calls == 1:
                return "risky-output: mitigation owners are missing"
            return "stable-output: risks include owners and mitigations"

        if current_step == 3:
            return "Final recommendation: proceed in phases with owner-based mitigations and guardrails."

        return "fallback"


class SafetyChecker(ReplanCheckerBase):
    async def evaluate(self, checker_input: ReplanCheckerInput, context: Any) -> ReplanVerdict:
        _ = context
        if "risky" in checker_input.step_result:
            return ReplanVerdict(
                action=ReplanAction.RETRY_CURRENT_STEP,
                reason="Safety checker rejected risky output.",
                confidence=0.95,
                regeneration_hints=["Remove risky language and add clear mitigation owners."],
            )
        return ReplanVerdict(action=ReplanAction.CONTINUE, reason="Safety checker approved.")


class StyleChecker(ReplanCheckerBase):
    async def evaluate(self, checker_input: ReplanCheckerInput, context: Any) -> ReplanVerdict:
        _ = context
        if "risky" in checker_input.step_result:
            return ReplanVerdict(
                action=ReplanAction.RETRY_CURRENT_STEP,
                reason="Style checker requests clearer structure.",
                confidence=0.77,
            )
        return ReplanVerdict(action=ReplanAction.CONTINUE, reason="Style checker approved.")


class CostChecker(ReplanCheckerBase):
    async def evaluate(self, checker_input: ReplanCheckerInput, context: Any) -> ReplanVerdict:
        _ = checker_input, context
        return ReplanVerdict(action=ReplanAction.CONTINUE, reason="Cost checker prefers continue.")


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
            aim="Establish rollout scope and constraints",
            reasoning_questions="Which systems and controls matter for launch?",
            stage_action="Produce concise framing",
            example_reasoning="Highlight only launch-critical scope",
            llm_config=LLMStepConfig(execution_mode=ExecutionMode.FAST),
        ),
        LLMStepDescription(
            number=2,
            title="Council Draft",
            aim="Generate draft for council validation",
            reasoning_questions="Is the draft safe, structured, and actionable?",
            stage_action="Write draft to be reviewed by checkers",
            example_reasoning="Explicitly mention risk ownership",
            dependencies=[1],
            checkpoint=True,
            checkpoint_name="council_phase",
            llm_config=LLMStepConfig(execution_mode=ExecutionMode.FAST),
        ),
        LLMStepDescription(
            number=3,
            title="Final Synthesis",
            aim="Finalize launch recommendation",
            reasoning_questions="Does the final recommendation reflect council feedback?",
            stage_action="Synthesize final narrative",
            example_reasoning="Keep recommendations practical",
            dependencies=[2],
            llm_config=LLMStepConfig(execution_mode=ExecutionMode.SELF_CRITIC),
        ),
    ]

    policy = ReplanPolicy(
        enabled=True,
        checkers=[
            RegisteredReplanCheckerConfig(name="safety"),
            RegisteredReplanCheckerConfig(name="style"),
            RegisteredReplanCheckerConfig(name="cost"),
        ],
        aggregation=ReplanAggregationConfig(
            strategy=ReplanAggregationStrategy.MANDATORY_PLUS_K_OF_REST,
            mandatory_checkers=["safety"],
            k=1,
        ),
    )

    return ReasoningChain(steps=steps, max_workers=1, replan_policy=policy)


def print_chain_overview(chain: ReasoningChain) -> None:
    _section("Chain Overview")
    header = f"{'Step':<6} {'Mode':<12} {'Deps':<10} {'Checkpoint':<16} {'Title'}"
    print(header)
    print("-" * len(header))
    for step in chain.steps:
        mode = step.llm_config.execution_mode.value if step.llm_config else "fast"
        deps = ",".join(str(dep) for dep in step.dependencies) if step.dependencies else "-"
        checkpoint = step.checkpoint_name if step.checkpoint else "-"
        print(f"{step.number:<6} {mode:<12} {deps:<10} {checkpoint:<16} {step.title}")


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
            f"votes={event.aggregation.trigger_count}/{event.aggregation.total_count} "
            f"mandatory_ok={event.aggregation.mandatory_satisfied} "
            f"triggered_by={event.triggering_checkers}"
        )


def main() -> None:
    print("CARL RE-PLAN Voting Council Trajectory Example")
    print("=" * 92)

    chain = build_chain()
    context = ReasoningContext(
        outer_context=SAMPLE_CONTEXT.strip(),
        api=CouncilTrajectoryClient(),
        model="unused",
        language=Language.ENGLISH,
    )
    context.register_replan_checker("safety", SafetyChecker())
    context.register_replan_checker("style", StyleChecker())
    context.register_replan_checker("cost", CostChecker())

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
