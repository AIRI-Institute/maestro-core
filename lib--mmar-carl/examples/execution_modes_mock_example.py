#!/usr/bin/env python3
"""
Example: Production execution modes with local mock client.

This example runs without API keys and demonstrates:
1) FAST forward pass
2) SELF_CRITIC with default LLM evaluator
3) SELF_CRITIC with evaluator chain (all must approve)

Usage:
    python examples/execution_modes_mock_example.py
"""

import sys
from pathlib import Path
from typing import Any

# Make `src.*` imports work when running as:
# `python examples/execution_modes_mock_example.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mmar_carl import (  # noqa: E402
    ExecutionMode,
    LLMClientBase,
    LLMStepConfig,
    LLMStepDescription,
    Language,
    ReasoningChain,
    ReasoningContext,
    SelfCriticDecision,
    SelfCriticEvaluatorBase,
)


class ModeDemoMockClient(LLMClientBase):
    """Mock client with deterministic responses for generation and LLM self-critic."""

    def __init__(self):
        self.calls: list[str] = []

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt, retries=1)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        _ = retries
        self.calls.append(prompt)
        lowered = prompt.lower()

        # Default llm self-critic evaluator prompt
        if "strict reviewer of an llm answer" in lowered:
            if "mock-draft-v2" in lowered or "mock-final-v2" in lowered:
                return '{"verdict":"APPROVE","review":"Looks complete."}'
            return '{"verdict":"DISAPPROVE","review":"Missing required details."}'

        # Regeneration prompts created by SELF_CRITIC flow
        if "regenerate the same task output with higher quality" in lowered:
            if "step 3" in lowered or "final" in lowered:
                return "mock-final-v2 with enough details and risk mitigation."
            return "mock-draft-v2 with clearer explanation."

        # Initial generation by step title markers in prompt body
        if "step 3. self_critic with evaluator chain" in lowered:
            return "mock-final-v1"
        if "step 2. self_critic with default llm evaluator" in lowered:
            return "mock-draft-v1"
        return "mock-fast-v1"


class KeywordGuardEvaluator(SelfCriticEvaluatorBase):
    """Non-LLM evaluator: requires specific keyword in candidate."""

    def __init__(self, required_keyword: str):
        self.required_keyword = required_keyword.lower()

    async def evaluate(
        self,
        step: Any,
        candidate: str,
        base_prompt: str,
        context: Any,
        llm_client: Any,
        retries: int,
    ) -> SelfCriticDecision:
        _ = step, base_prompt, context, llm_client, retries
        if self.required_keyword in candidate.lower():
            return SelfCriticDecision(
                verdict="APPROVE",
                review_text=f"Required keyword '{self.required_keyword}' found.",
                metadata={"llm_calls": 0},
            )
        return SelfCriticDecision(
            verdict="DISAPPROVE",
            review_text=f"Missing required keyword '{self.required_keyword}'.",
            metadata={"llm_calls": 0},
        )


def build_chain() -> ReasoningChain:
    """Build a three-step chain using FAST + SELF_CRITIC modes."""
    steps = [
        LLMStepDescription(
            number=1,
            title="FAST forward pass",
            aim="Generate initial draft quickly",
            reasoning_questions="Can we produce a concise first version?",
            stage_action="One-shot generation",
            example_reasoning="Fast mode is a strict single pass",
            llm_config=LLMStepConfig(execution_mode=ExecutionMode.FAST),
        ),
        LLMStepDescription(
            number=2,
            title="SELF_CRITIC with default LLM evaluator",
            aim="Improve draft quality with LLM review",
            reasoning_questions="Does default evaluator approve this output?",
            stage_action="Critic and regenerate if disapproved",
            example_reasoning="Expected one disapprove then approve",
            dependencies=[1],
            llm_config=LLMStepConfig(
                execution_mode=ExecutionMode.SELF_CRITIC,
                self_critic_evaluators=["llm"],
                self_critic_max_revisions=1,
            ),
        ),
        LLMStepDescription(
            number=3,
            title="SELF_CRITIC with evaluator chain",
            aim="Finalize output with multiple evaluators",
            reasoning_questions="Do all evaluators approve?",
            stage_action="Run llm + keyword_guard evaluators",
            example_reasoning="All evaluators must approve",
            dependencies=[2],
            llm_config=LLMStepConfig(
                execution_mode=ExecutionMode.SELF_CRITIC,
                self_critic_evaluators=["llm", "keyword_guard"],
                self_critic_max_revisions=2,
            ),
        ),
    ]
    return ReasoningChain(steps=steps, max_workers=1)


def print_mode_details(context: ReasoningContext) -> None:
    """Print per-step mode metadata summary."""
    details: dict[str, Any] = context.metadata.get("execution_mode_details", {})
    print("\nExecution mode details:")
    for step_number in sorted(details.keys(), key=int):
        info = details[step_number]
        print(
            f"  Step {step_number}: mode={info.get('execution_mode')} "
            f"llm_calls={info.get('llm_calls')} rounds={info.get('rounds')}"
        )
        if info.get("quality_warning"):
            print(f"    Warning: {info['quality_warning']}")


def main() -> None:
    """Run mock demo."""
    client = ModeDemoMockClient()
    context = ReasoningContext(
        outer_context="Mock task context",
        api=client,
        model="unused",
        language=Language.ENGLISH,
    )
    context.register_self_critic_evaluator("keyword_guard", KeywordGuardEvaluator(required_keyword="mitigation"))

    chain = build_chain()
    result = chain.execute(context)

    print(f"Success: {result.success}")
    print(f"Final output: {result.get_final_output()}")
    print(f"Total mock LLM calls: {len(client.calls)}")
    print_mode_details(context)


if __name__ == "__main__":
    main()
