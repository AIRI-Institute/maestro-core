#!/usr/bin/env python3
"""
Example: LLM Council — Multiple Models Voting on a Decision.

This example demonstrates how to use different LLM models within a single
CARL chain to implement a "council" pattern where multiple models independently
analyze a problem and then vote on the best decision.

Pattern overview:
    1. Multiple LLM steps run IN PARALLEL, each using a different model
       (via per-step LLMStepConfig overrides)
    2. A TOOL step aggregates the votes from all council members
    3. A final LLM step synthesizes the council's reasoning into a verdict

This is useful for:
    - Reducing bias from a single model
    - Getting diverse perspectives on complex decisions
    - Increasing confidence in critical choices
    - Leveraging different model strengths (cost vs quality vs speed)

Prerequisites:
    pip install 'mmar-carl[openai]'
    # or: pip install openai

Environment variables:
    OPENAI_API_KEY: Your OpenRouter API key (https://openrouter.ai)

Usage:
    export OPENAI_API_KEY="sk-or-v1-..."
    python examples/llm_council_example.py
"""

import asyncio
import json
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from mmar_carl import (
    Language,
    LLMStepConfig,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    StructuredOutputStepConfig,
    StructuredOutputStepDescription,
    ToolStepConfig,
    ToolStepDescription,
    create_openai_client,
)
from examples.utils import format_status


# ============================================================================
# Structured output model for the final verdict
# ============================================================================


class CouncilVerdict(BaseModel):
    """Structured result of the council's decision."""

    chosen_option: Literal["A", "B", "C"] = Field(..., description="The winning option letter: A, B, or C")
    option_title: str = Field(..., description="Short title of the chosen option, e.g. 'Go + PostgreSQL + Kubernetes'")
    consensus_level: Literal["unanimous", "majority", "split"] = Field(
        ..., description="How strongly the council agreed"
    )
    key_reason: str = Field(..., description="One-sentence summary of why this option was chosen")


# ============================================================================
# Sample decision scenario
# ============================================================================

DECISION_SCENARIO = """
Technology Stack Decision for a New Fintech Startup:

Context:
- Building a real-time payment processing platform
- Expected to handle 10,000 transactions per second at peak
- Team of 8 engineers (5 backend, 2 frontend, 1 DevOps)
- Must comply with PCI DSS and SOC 2 requirements
- MVP needed in 4 months, full launch in 8 months
- Budget: $500K for first year infrastructure

Options under consideration:

Option A: Go + PostgreSQL + Kubernetes
- Pros: High performance, strong concurrency, growing ecosystem
- Cons: Smaller talent pool, less mature ORM tooling

Option B: Java/Spring Boot + PostgreSQL + Kubernetes
- Pros: Battle-tested in fintech, vast ecosystem, strong typing
- Cons: Verbose, slower development speed, higher memory usage

Option C: Node.js/TypeScript + PostgreSQL + Kubernetes
- Pros: Full-stack JS, fast prototyping, large talent pool
- Cons: Single-threaded (needs clustering), less suitable for CPU-heavy tasks

Please evaluate each option and vote for the best choice.
"""


# ============================================================================
# Council configuration — models and their roles
# ============================================================================

# Each council member uses a different model with a tailored persona
# Models are configurable via DEFAULT_EXAMPLES_MODEL_1/2/3 environment variables
COUNCIL_MEMBERS = [
    {
        "number": 1,
        "title": "Council Member: Architecture Expert",
        "model": os.environ.get("DEFAULT_EXAMPLES_MODEL_1", "minimax/minimax-m2.5"),
        "temperature": 0.4,
        "system_role": "senior software architect with 20 years of experience in distributed systems",
        "focus": "system architecture, scalability, and long-term maintainability",
    },
    {
        "number": 2,
        "title": "Council Member: Startup CTO",
        "model": os.environ.get("DEFAULT_EXAMPLES_MODEL_2", "openai/gpt-4o"),
        "temperature": 0.5,
        "system_role": "startup CTO who has built and scaled 3 fintech companies",
        "focus": "time-to-market, team productivity, and hiring",
    },
    {
        "number": 3,
        "title": "Council Member: Security & Compliance Lead",
        "model": os.environ.get("DEFAULT_EXAMPLES_MODEL_3", "qwen/qwen3.5-plus-02-15"),
        "temperature": 0.3,
        "system_role": "security engineer specializing in PCI DSS compliance for payment systems",
        "focus": "security, compliance requirements, and risk assessment",
    },
]


# ============================================================================
# Vote aggregation tool
# ============================================================================


def aggregate_council_votes(
    vote_1: str,
    vote_2: str,
    vote_3: str,
) -> dict[str, Any]:
    """
    Aggregate votes from council members.

    Parses each council member's response to extract their vote (Option A/B/C)
    and builds a summary of the council's decision.
    """
    votes = {"vote_1": vote_1, "vote_2": vote_2, "vote_3": vote_3}
    labels = {
        "vote_1": COUNCIL_MEMBERS[0]["title"],
        "vote_2": COUNCIL_MEMBERS[1]["title"],
        "vote_3": COUNCIL_MEMBERS[2]["title"],
    }
    models = {
        "vote_1": COUNCIL_MEMBERS[0]["model"],
        "vote_2": COUNCIL_MEMBERS[1]["model"],
        "vote_3": COUNCIL_MEMBERS[2]["model"],
    }

    # Extract votes — look for "Option A", "Option B", or "Option C"
    vote_pattern = re.compile(r"Option\s+([ABC])", re.IGNORECASE)
    extracted = {}
    tally: dict[str, list[str]] = {}

    for key, response in votes.items():
        matches = vote_pattern.findall(response)
        if matches:
            # Take the last mentioned option as the final vote
            choice = f"Option {matches[-1].upper()}"
        else:
            choice = "Abstain"

        extracted[key] = {
            "member": labels[key],
            "model": models[key],
            "vote": choice,
            "reasoning_excerpt": response[:300] + "..." if len(response) > 300 else response,
        }

        tally.setdefault(choice, [])
        tally[choice].append(labels[key])

    # Determine winner
    winner = max(tally, key=lambda k: len(tally[k])) if tally else "No consensus"
    unanimous = len(tally) == 1 and "Abstain" not in tally

    return {
        "votes": extracted,
        "tally": {k: len(v) for k, v in tally.items()},
        "winner": winner,
        "unanimous": unanimous,
        "consensus_level": "unanimous"
        if unanimous
        else ("majority" if any(len(v) >= 2 for v in tally.values()) else "split"),
    }


# ============================================================================
# Example 1: Full LLM Council with parallel voting
# ============================================================================


async def example_llm_council():
    """
    Example 1: LLM Council — Parallel Multi-Model Voting.

    Three different LLM models independently evaluate a technology decision,
    then a tool step aggregates votes, and a final LLM synthesizes the verdict.

    Chain structure (DAG):
        [Member 1] ──┐
        [Member 2] ──┼── [Aggregate Votes] ── [Final Verdict] ── [Structured Result]
        [Member 3] ──┘

    Steps 1-3 run in parallel (different models), step 4 aggregates,
    step 5 produces the final synthesis, step 6 extracts a structured verdict.
    """
    print("\n" + "=" * 60)
    print("Example 1: LLM Council — Parallel Multi-Model Voting")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Skipping: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='sk-or-v1-...'")
        return

    # Build council member steps — each uses a different model via LLMStepConfig
    council_steps: list[LLMStepDescription | ToolStepDescription | StructuredOutputStepDescription] = []

    for member in COUNCIL_MEMBERS:
        step = LLMStepDescription(
            number=member["number"],
            title=member["title"],
            aim=(
                f"Evaluate the technology options from the perspective of a {member['system_role']}. "
                f"Focus on {member['focus']}. "
                "You MUST end your response with a clear vote: 'My vote: Option X' where X is A, B, or C."
            ),
            reasoning_questions=(
                "Which option best fits the requirements? "
                "What are the critical trade-offs? "
                "What risks does each option carry?"
            ),
            stage_action=(
                f"Analyze all three options through the lens of {member['focus']} and cast a definitive vote."
            ),
            example_reasoning=(
                "After evaluating scalability, team expertise, and compliance needs, "
                "Option X provides the best balance. My vote: Option X"
            ),
            step_context_queries=["transaction", "compliance", "team", "budget"],
            # Per-step model override — this is the key feature!
            llm_config=LLMStepConfig(
                model=member["model"],
                temperature=member["temperature"],
                max_tokens=2048,
            ),
            # No dependencies — all council members vote independently and in parallel
            dependencies=[],
        )
        council_steps.append(step)

    # Step 4: Tool step — aggregate votes from all council members
    aggregate_step = ToolStepDescription(
        number=4,
        title="Vote Aggregation",
        config=ToolStepConfig(
            tool_name="aggregate_council_votes",
            input_mapping={
                "vote_1": "$metadata.step_1",
                "vote_2": "$metadata.step_2",
                "vote_3": "$metadata.step_3",
            },
        ),
        dependencies=[1, 2, 3],  # Wait for all council members
    )
    council_steps.append(aggregate_step)

    # Step 5: Final synthesis by a capable model
    synthesis_step = LLMStepDescription(
        number=5,
        title="Council Verdict",
        aim=(
            "Synthesize the council's votes and reasoning into a final verdict. "
            "Present the winning option, explain why the council chose it, "
            "and note any dissenting opinions or concerns raised."
        ),
        reasoning_questions=(
            "What was the council's decision? Was it unanimous or split? "
            "What were the strongest arguments for and against the winning option? "
            "What conditions or caveats should be noted?"
        ),
        stage_action="Produce a clear, structured final verdict based on the council's deliberation",
        example_reasoning=(
            "The council voted 2-1 in favor of Option B. "
            "The architecture expert and CTO agreed on its maturity, "
            "while the security lead preferred Option A for its performance characteristics."
        ),
        llm_config=LLMStepConfig(
            model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "minimax/minimax-m2.5"),
            temperature=0.3,
            max_tokens=3000,
        ),
        dependencies=[4],
    )
    council_steps.append(synthesis_step)

    # Step 6: Structured output — extract the winning option letter
    structured_step = StructuredOutputStepDescription(
        number=6,
        title="Structured Verdict",
        dependencies=[5],
        config=StructuredOutputStepConfig.from_pydantic_model(
            CouncilVerdict,
            input_source="$history[-1]",
            instruction=(
                "Extract the council's final decision from the verdict above. "
                "Return the chosen option letter (A, B, or C), its short title, "
                "the consensus level, and one-sentence key reason."
            ),
            strict_json=True,
        ),
    )
    council_steps.append(structured_step)

    # Create the chain with parallel workers for council members
    chain = ReasoningChain(
        steps=council_steps,
        max_workers=3,  # All 3 council members can run in parallel
        trace_name="LLM Council Parallel Multi-Model Voting",
    )

    print("\nCouncil members:")
    for member in COUNCIL_MEMBERS:
        print(f"  - {member['title']} ({member['model']})")
    print(f"\nChain: {len(chain.steps)} steps, max_workers=3")
    print(f"Execution plan: {chain.get_execution_plan()}")

    # Create base client (its model will be overridden per-step by LLMStepConfig)
    client = create_openai_client(
        api_key=api_key,
        model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"),
        extra_headers={
            "HTTP-Referer": "https://github.com/mmar-workflow/carl-experiments",
            "X-Title": "CARL LLM Council Example",
        },
    )

    context = ReasoningContext(
        outer_context=DECISION_SCENARIO,
        api=client,
        model="unused",
        language=Language.ENGLISH,
        system_prompt=(
            "You are participating in an expert council to evaluate technology decisions. "
            "Provide thorough analysis and a clear vote."
        ),
    )

    # Register the vote aggregation tool
    context.register_tool("aggregate_council_votes", aggregate_council_votes)

    # Execute the council
    print("\nExecuting council deliberation...")
    result = await chain.execute_async(context)

    print(f"\nExecution: {format_status(result.success)}")
    print(f"Time: {result.total_execution_time:.2f}s")

    if result.success:
        # Show individual votes
        aggregate_result = result.step_results[3]  # Step 4 (0-indexed)
        if aggregate_result.success and aggregate_result.result_data:
            print("\n" + "-" * 40)
            print("VOTE RESULTS:")
            print("-" * 40)
            metadata = aggregate_result.result_data
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    pass

            if isinstance(metadata, dict):
                tally = metadata.get("tally", {})
                winner = metadata.get("winner", "Unknown")
                consensus = metadata.get("consensus_level", "Unknown")
                print(f"  Winner: {winner}")
                print(f"  Consensus: {consensus}")
                print(f"  Tally: {tally}")

                votes = metadata.get("votes", {})
                for key, vote_info in votes.items():
                    if isinstance(vote_info, dict):
                        print(f"\n  {vote_info.get('member', key)}:")
                        print(f"    Model: {vote_info.get('model', 'N/A')}")
                        print(f"    Vote:  {vote_info.get('vote', 'N/A')}")

        # Show final verdict (step 5 — free-form synthesis)
        verdict_result = result.step_results[4]  # Step 5 (0-indexed)
        if verdict_result.success:
            print("\n" + "-" * 40)
            print("FINAL VERDICT:")
            print("-" * 40)
            verdict_text = verdict_result.result or ""
            print(verdict_text[:1500] + "..." if len(verdict_text) > 1500 else verdict_text)

        # Show structured result (step 6 — machine-readable option letter)
        structured_result = result.step_results[5]  # Step 6 (0-indexed)
        if structured_result.success and structured_result.result_data:
            print("\n" + "=" * 40)
            print("STRUCTURED RESULT:")
            print("=" * 40)
            try:
                verdict = CouncilVerdict.model_validate(structured_result.result_data)
                print(f"  Chosen Option  : {verdict.chosen_option}")
                print(f"  Option Title   : {verdict.option_title}")
                print(f"  Consensus      : {verdict.consensus_level}")
                print(f"  Key Reason     : {verdict.key_reason}")
            except Exception as e:
                print(f"  Validation failed: {e}")
                print(f"  Raw data: {json.dumps(structured_result.result_data, indent=2, ensure_ascii=False)}")
    else:
        print("\nErrors:")
        for step_result in result.step_results:
            if not step_result.success:
                print(f"  Step {step_result.step_number} ({step_result.step_title}): {step_result.error_message}")


# ============================================================================
# Example 2: Lightweight council with ChainBuilder
# ============================================================================


async def example_quick_council():
    """
    Example 2: Quick Council using ChainBuilder.

    A simpler council pattern where two models provide opinions and a third
    synthesizes. Uses ChainBuilder for concise chain construction.

    This pattern is useful when you want a quick second opinion from a
    different model without a full voting mechanism.
    """
    print("\n" + "=" * 60)
    print("Example 2: Quick Two-Model Review")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Skipping: OPENAI_API_KEY not set")
        return

    question = """
Should we use microservices or a monolith architecture for our new e-commerce platform?

Context:
- Small team (4 developers)
- Need to launch in 3 months
- Expected 1000 daily active users initially
- Plans to scale to 100K users within 2 years
"""

    steps = [
        # Model 1: Fast, practical perspective
        LLMStepDescription(
            number=1,
            title="Pragmatic Advisor",
            aim=(
                "Provide a practical recommendation focusing on speed of delivery "
                "and team size constraints. End with 'My recommendation: [Monolith/Microservices]'"
            ),
            reasoning_questions="What is the most practical choice given the team size and timeline?",
            stage_action="Evaluate from a delivery and team productivity standpoint",
            example_reasoning="With 4 devs and 3 months, a monolith is faster to ship",
            llm_config=LLMStepConfig(
                model=os.environ.get("DEFAULT_EXAMPLES_MODEL_1", "minimax/minimax-m2.5"),
                temperature=0.4,
            ),
        ),
        # Model 2: Thorough, strategic perspective
        LLMStepDescription(
            number=2,
            title="Strategic Advisor",
            aim=(
                "Provide a strategic recommendation focusing on long-term scalability "
                "and architecture evolution. End with 'My recommendation: [Monolith/Microservices]'"
            ),
            reasoning_questions="What architecture best supports growth from 1K to 100K users?",
            stage_action="Evaluate from a scalability and future growth standpoint",
            example_reasoning="Starting monolith with clear module boundaries enables future split",
            llm_config=LLMStepConfig(
                model=os.environ.get("DEFAULT_EXAMPLES_MODEL_3", "openai/gpt-4o"),
                temperature=0.5,
            ),
        ),
        # Synthesis step: combine both perspectives
        LLMStepDescription(
            number=3,
            title="Final Recommendation",
            aim=(
                "Review both advisors' recommendations and produce a final, balanced verdict. "
                "Highlight where they agree, where they differ, and give a clear final recommendation."
            ),
            reasoning_questions=(
                "Do the advisors agree or disagree? What is the best recommendation considering both perspectives?"
            ),
            stage_action="Synthesize both viewpoints into a single actionable recommendation",
            example_reasoning=("Both advisors agree on starting with a monolith but differ on migration timing"),
            llm_config=LLMStepConfig(
                model=os.environ.get("DEFAULT_EXAMPLES_MODEL_3", "qwen/qwen3.5-plus-02-15"),
                temperature=0.3,
            ),
            dependencies=[1, 2],
        ),
    ]

    chain = ReasoningChain(steps=steps, max_workers=2, trace_name="Quick Two-Model Review")

    print("Advisors:")
    print(f"  1. Pragmatic Advisor  ({steps[0].llm_config.model})")
    print(f"  2. Strategic Advisor  ({steps[1].llm_config.model})")
    print(f"  3. Final Synthesis    ({steps[2].llm_config.model})")
    print(f"\nExecution plan: {chain.get_execution_plan()}")
    print("  Steps 1 and 2 run in parallel, step 3 waits for both")

    client = create_openai_client(
        api_key=api_key,
        model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"),
    )

    context = ReasoningContext(
        outer_context=question,
        api=client,
        model="unused",
        language=Language.ENGLISH,
        system_prompt="You are a senior technology advisor. Be direct and actionable.",
    )

    print("\nExecuting quick council...")
    result = await chain.execute_async(context)

    print(f"\nExecution: {format_status(result.success)}")
    print(f"Time: {result.total_execution_time:.2f}s")

    if result.success:
        # Show each advisor's response
        for i, step_result in enumerate(result.step_results):
            if step_result.success and i < 2:  # First two are advisors
                print(f"\n{'~' * 40}")
                print(f"{steps[i].title} ({steps[i].llm_config.model}):")
                print("~" * 40)
                output = step_result.result or ""
                print(output[:400] + "..." if len(output) > 400 else output)

        # Show final recommendation
        print(f"\n{'=' * 40}")
        print("FINAL RECOMMENDATION:")
        print("=" * 40)
        print(result.get_final_output())


# ============================================================================
# Main
# ============================================================================


async def main():
    """Run all LLM council examples."""
    print("CARL LLM Council Examples")
    print("=" * 60)
    print("\nThese examples demonstrate the 'LLM Council' pattern:")
    print("multiple models vote independently, then results are aggregated.")
    print("\nRequires OPENAI_API_KEY for execution.")

    if not os.environ.get("OPENAI_API_KEY"):
        print("\nWARNING: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='sk-or-v1-...'")
        print("Get your key at: https://openrouter.ai/keys")
    else:
        print("\nOPENAI_API_KEY is set")

    await example_llm_council()
    await example_quick_council()

    print("\n" + "=" * 60)
    print("LLM Council examples completed!")
    print("=" * 60)
    print("\nKey patterns demonstrated:")
    print("  - Per-step model overrides via LLMStepConfig")
    print("  - Parallel execution of independent council members")
    print("  - Tool-based vote aggregation")
    print("  - Final synthesis from council deliberation")
    print("\nNext steps:")
    print("  - See basic_chain_example.py for chain fundamentals")
    print("  - See openrouter_example.py for more API options")
    print("  - See tool_steps_example.py for tool/memory integration")


if __name__ == "__main__":
    asyncio.run(main())
