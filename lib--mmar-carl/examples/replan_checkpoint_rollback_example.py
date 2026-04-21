#!/usr/bin/env python3
"""Checkpoint rollback example for RE-PLAN policy."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.utils import format_status  # noqa: E402
from mmar_carl import (  # noqa: E402
    LLMClientBase,
    ReplanAction,
    ReplanPolicy,
    ReplanRollbackTarget,
    ReplanTargetType,
    ReasoningContext,
    RuleBasedReplanCheckerConfig,
)
from mmar_carl.chain import ChainBuilder  # noqa: E402


class NoopLLMClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        _ = prompt
        return ""

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        _ = prompt, retries
        return ""


def main() -> None:
    calls = {"checkpoint": 0, "flaky": 0, "final": 0}

    def checkpoint_tool() -> str:
        calls["checkpoint"] += 1
        return "checkpoint-ready"

    def flaky_tool() -> str:
        calls["flaky"] += 1
        return "bad" if calls["flaky"] == 1 else "good"

    def finalize_tool(value: str) -> str:
        calls["final"] += 1
        return f"final:{value}"

    policy = ReplanPolicy(
        enabled=True,
        checkers=[
            RuleBasedReplanCheckerConfig(
                name="rollback_on_bad",
                result_substrings=["bad"],
                action_on_match=ReplanAction.REPLAN_FROM_CHECKPOINT,
                rollback_target_on_match=ReplanRollbackTarget(
                    target_type=ReplanTargetType.NAMED_CHECKPOINT,
                    checkpoint_name="cp_validation",
                ),
                feedback_on_match=["Retry the flaky step with corrected assumptions."],
            )
        ],
    )

    chain = (
        ChainBuilder()
        .add_tool_step(
            number=1, title="Checkpoint", tool_name="checkpoint", checkpoint=True, checkpoint_name="cp_validation"
        )
        .add_tool_step(number=2, title="Flaky step", tool_name="flaky", dependencies=[1])
        .add_tool_step(
            number=3,
            title="Finalize",
            tool_name="finalize",
            dependencies=[2],
            input_mapping={"value": "$steps.2.result"},
        )
        .with_replan_policy(policy)
        .build()
    )

    context = ReasoningContext(outer_context="input", api=NoopLLMClient(), model="unused")
    context.register_tool("checkpoint", checkpoint_tool)
    context.register_tool("flaky", flaky_tool)
    context.register_tool("finalize", finalize_tool)

    result = chain.execute(context)

    print("Checkpoint Rollback RE-PLAN Example")
    print("=" * 64)
    print(f"Success: {format_status(result.success)}")
    print(f"Final output: {result.get_final_output()}")
    print(f"Tool calls: {calls}")
    for event in result.replan_events:
        print(f"  step={event.step_number} action={event.final_action} target={event.rollback_target}")


if __name__ == "__main__":
    main()
