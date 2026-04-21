#!/usr/bin/env python3
"""
Example: Production execution modes pipeline (FAST + SELF_CRITIC).

This example demonstrates:
1) FAST mode (single forward pass)
2) SELF_CRITIC mode with default LLM evaluator ("llm")
3) SELF_CRITIC mode with evaluator chain (all must approve)

Prerequisites:
    pip install 'mmar-carl[openai]'

Environment variables:
    OPENAI_API_KEY: API key for your provider
    EXECUTION_MODES_MODEL: Optional model override (default: Sber/GigaChat-Max-V2)
    EXECUTION_MODES_TEMPERATURE: Optional temperature override (default: 0.87)
    OPENAI_BASE_URL: Optional provider base URL (OpenRouter/local proxy/etc.)
    OPENAI_SSL_VERIFY: Optional TLS verification toggle (default: true)

Usage:
    export OPENAI_API_KEY="sk-or-v1-..."
    python examples/execution_modes_pipeline_example.py
"""

import os
import sys
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Make `src.*` and `examples.*` imports work when running as:
# `python examples/execution_modes_pipeline_example.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.utils import format_status  # noqa: E402
from mmar_carl import (  # noqa: E402
    ExecutionMode,
    LLMStepConfig,
    LLMStepDescription,
    Language,
    ReasoningChain,
    ReasoningContext,
    SelfCriticDecision,
    SelfCriticEvaluatorBase,
    create_openai_client,
)

SAMPLE_CONTEXT = """
Issue: Build a concise release note for an internal analytics tool.
Must include:
- what changed
- why it matters
- one risk
- one mitigation
Tone: practical and short.
"""


class MinLengthSelfCriticEvaluator(SelfCriticEvaluatorBase):
    """Simple non-LLM evaluator that enforces minimum response length."""

    def __init__(self, min_chars: int = 180):
        self.min_chars = min_chars

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
        if len(candidate.strip()) >= self.min_chars:
            return SelfCriticDecision(
                verdict="APPROVE",
                review_text=f"Length is sufficient (>= {self.min_chars} chars).",
                metadata={"llm_calls": 0},
            )
        return SelfCriticDecision(
            verdict="DISAPPROVE",
            review_text=f"Response is too short; expected at least {self.min_chars} chars.",
            metadata={"llm_calls": 0},
        )


def build_chain() -> ReasoningChain:
    """Create a 3-step chain for FAST + SELF_CRITIC production flow."""
    steps = [
        LLMStepDescription(
            number=1,
            title="FAST Draft",
            aim="Create a short draft release note",
            reasoning_questions="What changed and why does it matter?",
            stage_action="Write concise draft",
            example_reasoning="Focus only on required bullet points",
            step_context_queries=["what changed", "why it matters"],
            llm_config=LLMStepConfig(execution_mode=ExecutionMode.FAST),
        ),
        LLMStepDescription(
            number=2,
            title="SELF_CRITIC with Default LLM Evaluator",
            aim="Improve quality of release note",
            reasoning_questions="Is the draft complete, clear, and concise?",
            stage_action="Run LLM evaluator and regenerate on disapproval",
            example_reasoning="If disapproved, regenerate once with review notes",
            dependencies=[1],
            llm_config=LLMStepConfig(
                execution_mode=ExecutionMode.SELF_CRITIC,
                self_critic_evaluators=["llm"],
                self_critic_max_revisions=1,
            ),
        ),
        LLMStepDescription(
            number=3,
            title="SELF_CRITIC with Evaluator Chain",
            aim="Produce final release note with stronger quality checks",
            reasoning_questions="Do all evaluators approve this response?",
            stage_action="Use evaluator chain (all-must-approve)",
            example_reasoning="LLM evaluator + min-length guard must both approve",
            dependencies=[2],
            llm_config=LLMStepConfig(
                execution_mode=ExecutionMode.SELF_CRITIC,
                self_critic_evaluators=["llm", "min_length_guard"],
                self_critic_max_revisions=2,
            ),
        ),
    ]
    return ReasoningChain(steps=steps, max_workers=1)


def _truncate(text: str, width: int) -> str:
    """Return text trimmed to width with ellipsis."""
    normalized = " ".join((text or "").split())
    if len(normalized) <= width:
        return normalized
    if width <= 3:
        return normalized[:width]
    return normalized[: width - 3] + "..."


def _section(title: str) -> None:
    """Print a visual section separator."""
    line = "=" * 92
    print(f"\n{line}\n{title}\n{line}")


def _env_bool(name: str, default: bool = True) -> bool:
    """Parse a boolean environment variable."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def print_runtime_config(model: str, temperature: float, base_url: str | None, verify_ssl: bool) -> None:
    """Print runtime model/provider settings."""
    _section("Runtime Configuration")
    print(f"{'Model':<18}: {model}")
    print(f"{'Temperature':<18}: {temperature:.2f}")
    print(f"{'Base URL':<18}: {base_url or '(default from library/env)'}")
    print(f"{'SSL verify':<18}: {verify_ssl}")
    print(f"{'Mode':<18}: FAST + SELF_CRITIC")


def print_chain_overview(chain: ReasoningChain) -> None:
    """Print chain structure and execution mode configuration."""
    _section("Chain Overview")
    header = f"{'Step':<6} {'Mode':<12} {'Deps':<10} {'Evaluators':<34} {'Rev':<4} {'Title'}"
    print(header)
    print("-" * len(header))
    for step in chain.steps:
        llm_config = step.llm_config or LLMStepConfig()
        mode = llm_config.execution_mode.value
        deps = ",".join(str(d) for d in step.dependencies) if step.dependencies else "-"
        evaluators = ",".join(llm_config.self_critic_evaluators) if mode == ExecutionMode.SELF_CRITIC.value else "-"
        revisions = llm_config.self_critic_max_revisions if mode == ExecutionMode.SELF_CRITIC.value else 0
        print(
            f"{step.number:<6} {mode:<12} {deps:<10} "
            f"{_truncate(evaluators, 34):<34} {revisions:<4} {_truncate(step.title, 24)}"
        )


def print_step_results(result: Any) -> None:
    """Print per-step execution outcomes with concise output previews."""
    _section("Step Results")
    header = f"{'Step':<6} {'Status':<10} {'Time':<8} {'Output / Error'}"
    print(header)
    print("-" * len(header))
    for step_result in sorted(result.step_results, key=lambda r: r.step_number):
        status = format_status(step_result.success)
        exec_time = f"{step_result.execution_time:.2f}s" if step_result.execution_time is not None else "n/a"
        preview = step_result.result if step_result.success else (step_result.error_message or "")
        print(f"{step_result.step_number:<6} {status:<10} {exec_time:<8} {_truncate(preview, 92)}")


def print_mode_details(context: ReasoningContext) -> None:
    """Print per-step execution mode details captured by the executor."""
    details: dict[str, Any] = context.metadata.get("execution_mode_details", {})
    if not details:
        _section("Execution Mode Details")
        print("No execution mode details found.")
        return

    _section("Execution Mode Details")
    header = f"{'Step':<6} {'Mode':<12} {'LLM Calls':<10} {'Rounds':<8} {'Policy':<22} {'Warning'}"
    print(header)
    print("-" * len(header))
    for step_number in sorted(details.keys(), key=int):
        step_details = details[step_number]
        mode = step_details.get("execution_mode", "unknown")
        llm_calls = step_details.get("llm_calls", 0)
        rounds = step_details.get("rounds", 0)
        evaluator_policy = step_details.get("evaluator_policy", "-")
        warning = _truncate(step_details.get("quality_warning", ""), 26)
        print(f"{step_number:<6} {mode:<12} {llm_calls:<10} {rounds:<8} {evaluator_policy:<22} {warning}")

        evaluator_decisions = step_details.get("evaluator_decisions", [])
        for round_info in evaluator_decisions:
            round_num = round_info.get("round")
            approved = round_info.get("approved")
            verdicts = ", ".join(f"{e.get('evaluator')}={e.get('verdict')}" for e in round_info.get("evaluators", []))
            print(f"  round {round_num}: approved={approved}; {verdicts}")


def print_connection_diagnostics(result: Any, model: str, base_url: str | None, verify_ssl: bool) -> None:
    """Print targeted diagnostics for common connection-related failures."""
    failed_steps = result.get_failed_steps()
    if not failed_steps:
        return

    all_errors = " ".join((step.error_message or "").lower() for step in failed_steps)
    markers = (
        "connection error",
        "connecterror",
        "ssl",
        "certificate",
        "timed out",
        "name or service not known",
        "nodename nor servname provided",
    )
    if not any(marker in all_errors for marker in markers):
        return

    _section("Connection Diagnostics")
    print(f"{'Base URL':<18}: {base_url or '(default from library/env)'}")
    print(f"{'Model':<18}: {model}")
    print(f"{'SSL verify':<18}: {verify_ssl}")
    print("test_inference.py uses `DefaultHttpxClient(verify=False)`.")
    if verify_ssl:
        print("Try OPENAI_SSL_VERIFY=false if your provider uses a custom/self-signed certificate.")
    else:
        print("SSL verification is already disabled; check DNS/network reachability and API key/provider pairing.")


def main() -> None:
    """Run the production execution modes pipeline example."""
    print("CARL Execution Modes Pipeline Example")
    print("=" * 92)
    print("This demo runs FAST + SELF_CRITIC steps and prints detailed execution diagnostics.")

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nSkipping: OPENAI_API_KEY not set")
        print('Set it with: export OPENAI_API_KEY="sk-or-v1-..."')
        return

    model = os.environ.get("DEFAULT_EXAMPLES_MODEL", "Openai/Gpt-oss-120b")
    temperature_raw = os.environ.get("EXECUTION_MODES_TEMPERATURE", "0.87")
    base_url = os.environ.get("OPENAI_BASE_URL")
    verify_ssl = _env_bool("OPENAI_SSL_VERIFY", default=True)
    try:
        temperature = float(temperature_raw)
    except ValueError:
        temperature = 0.87
        print(f"Warning: invalid EXECUTION_MODES_TEMPERATURE='{temperature_raw}', using {temperature}")

    client = create_openai_client(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        verify_ssl=verify_ssl,
    )
    print_runtime_config(model=model, temperature=temperature, base_url=base_url, verify_ssl=verify_ssl)

    context = ReasoningContext(
        outer_context=SAMPLE_CONTEXT.strip(),
        api=client,
        model="unused",
        language=Language.ENGLISH,
    )
    context.register_self_critic_evaluator("min_length_guard", MinLengthSelfCriticEvaluator(min_chars=180))

    chain = build_chain()
    print_chain_overview(chain)

    _section("Execution")
    result = chain.execute(context)

    print(f"\nOverall status: {format_status(result.success)}")
    if result.total_execution_time is not None:
        print(f"Total time: {result.total_execution_time:.2f}s")

    print_step_results(result)
    print_connection_diagnostics(result, model=model, base_url=base_url, verify_ssl=verify_ssl)

    _section("Final Output")
    print(_truncate(result.get_final_output(), 800))
    print_mode_details(context)


if __name__ == "__main__":
    main()
