#!/usr/bin/env python3
"""Budget / loop prevention RE-PLAN example."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.utils import format_status  # noqa: E402
from mmar_carl import (  # noqa: E402
    Language,
    LLMClientBase,
    LLMStepDescription,
    ReplanAction,
    ReplanBudgetConfig,
    ReplanPolicy,
    RuleBasedReplanCheckerConfig,
    ReasoningChain,
    ReasoningContext,
)


class LoopClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        _ = prompt
        return "loop-output"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        _ = prompt, retries
        return "loop-output"


def main() -> None:
    policy = ReplanPolicy(
        enabled=True,
        checkers=[
            RuleBasedReplanCheckerConfig(
                name="loop_guard",
                result_substrings=["loop-output"],
                action_on_match=ReplanAction.RETRY_CURRENT_STEP,
            )
        ],
        budgets=ReplanBudgetConfig(
            max_replans_per_chain=1,
            max_replans_per_step=1,
            max_visits_per_checkpoint=2,
            max_same_rollback_target_repeats=1,
            fail_on_budget_exhaustion=True,
        ),
    )

    chain = ReasoningChain(
        steps=[
            LLMStepDescription(
                number=1,
                title="Loop-prone step",
                aim="Generate answer",
                reasoning_questions="Answer clearly",
                stage_action="Generate",
                example_reasoning="Stay concise",
            )
        ],
        max_workers=1,
        replan_policy=policy,
    )

    context = ReasoningContext(outer_context="input", api=LoopClient(), model="unused", language=Language.ENGLISH)
    result = chain.execute(context)

    print("Budget Guard RE-PLAN Example")
    print("=" * 64)
    print(f"Success: {format_status(result.success)}")
    print(f"Failure reason: {result.metadata.get('replan', {}).get('failure_reason', '')}")
    print(f"Replan summary: {result.metadata.get('replan', {})}")
    for event in result.replan_events:
        print(f"  action={event.final_action} budget_exhausted={event.budget_exhausted} note={event.note}")


if __name__ == "__main__":
    main()
