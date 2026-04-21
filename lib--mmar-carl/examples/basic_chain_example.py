#!/usr/bin/env python3
"""
Example: Basic CARL chain usage with OpenRouter.

This example demonstrates the fundamental concepts of CARL:
- Creating reasoning steps with dependencies
- Building chains using both explicit steps and ChainBuilder
- Executing chains synchronously with OpenRouter
- Working with results and history

Prerequisites:
    pip install 'mmar-carl[openai]'
    # or: pip install openai

Environment variables:
    OPENAI_API_KEY: Your OpenRouter API key (https://openrouter.ai)

Usage:
    export OPENAI_API_KEY="sk-or-v1-..."
    python examples/basic_chain_example.py
"""

import asyncio
import os
import uuid
from dotenv import load_dotenv

from mmar_carl import (
    ChainBuilder,
    Language,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    StepDescription,
    create_openai_client,
    langfuse_flush,
)
from examples.utils import format_status


# Sample data for examples
SAMPLE_TEXT = """
Company XYZ Q4 2024 Report Summary:

Financial Highlights:
- Total Revenue: $2.5 million (up 15% YoY)
- Operating Expenses: $1.8 million
- Net Profit: $700,000
- EBITDA Margin: 32%

Key Achievements:
- Launched new product line generating $400K in revenue
- Expanded to 3 new markets
- Reduced customer acquisition cost by 20%

Challenges:
- Supply chain disruptions caused delays
- Increased competition in core market
- Rising labor costs
"""


def _env_bool(name: str, default: bool = True) -> bool:
    """Parse a boolean environment variable."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def _print_failure_diagnostics(result, client) -> None:
    """Print detailed error diagnostics for failed executions."""
    failed_steps = result.get_failed_steps()
    if not failed_steps:
        return

    print("\nFailed step details:")
    for step in failed_steps:
        error = step.error_message or "Unknown error"
        print(f"  - Step {step.step_number} ({step.step_title}): {error}")

    all_errors = " ".join((step.error_message or "").lower() for step in failed_steps)
    connection_markers = (
        "connection error",
        "connecterror",
        "ssl",
        "certificate",
        "timed out",
        "name or service not known",
        "nodename nor servname provided",
    )
    if not any(marker in all_errors for marker in connection_markers):
        return

    config = getattr(client, "config", None)
    base_url = getattr(config, "base_url", "(unknown)")
    model = getattr(config, "model", "(unknown)")
    verify_ssl = getattr(config, "verify_ssl", True)

    print("\nConnection diagnostics:")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    print(f"  SSL verify: {verify_ssl}")
    print("  Your test_inference.py uses `DefaultHttpxClient(verify=False)`.")
    if verify_ssl:
        print("  Try setting OPENAI_SSL_VERIFY=false for endpoints with custom/self-signed certificates.")
    else:
        print("  SSL verification is disabled already; check DNS/network reachability and API key/provider pairing.")


async def example_explicit_steps():
    """
    Example 1: Creating and executing a chain with explicit step definitions.

    This shows the traditional way of defining steps with full control
    over all parameters, then executing with OpenRouter.
    """
    print("\n" + "=" * 60)
    print("Example 1: Explicit Step Definitions + Execution")
    print("=" * 60)

    # Define steps explicitly using the new typed API
    steps = [
        LLMStepDescription(
            number=1,
            title="Financial Metrics Extraction",
            aim="Extract and organize key financial metrics from the report",
            reasoning_questions="What are the main financial figures? How do they compare YoY?",
            stage_action="Identify and list all financial metrics with their values",
            example_reasoning="Revenue of $2.5M with 15% growth indicates strong performance",
            step_context_queries=["Revenue", "Profit", "EBITDA"],
        ),
        LLMStepDescription(
            number=2,
            title="Achievement Analysis",
            aim="Analyze the key achievements and their business impact",
            reasoning_questions="What were the main achievements? What is their strategic value?",
            stage_action="Evaluate each achievement's contribution to business growth",
            example_reasoning="New product line contributes 16% of total revenue",
            step_context_queries=["Achievements", "product", "market"],
            dependencies=[1],  # Depends on financial context from step 1
        ),
        LLMStepDescription(
            number=3,
            title="Risk Assessment",
            aim="Identify and assess challenges and risks",
            reasoning_questions="What challenges exist? What is their potential impact?",
            stage_action="Analyze each challenge and estimate risk severity",
            example_reasoning="Supply chain issues may affect Q1 delivery targets",
            step_context_queries=["Challenges", "disruptions", "cost"],
            dependencies=[1],  # Can run in parallel with step 2
        ),
        LLMStepDescription(
            number=4,
            title="Executive Summary",
            aim="Synthesize findings into actionable executive summary",
            reasoning_questions="What are the key takeaways? What actions are recommended?",
            stage_action="Create concise summary with recommendations",
            example_reasoning="Strong financial performance despite challenges suggests focus on supply chain",
            dependencies=[2, 3],  # Depends on both analysis steps
        ),
    ]

    # Create chain with parallel execution enabled
    chain = ReasoningChain(
        steps=steps,
        max_workers=2,  # Steps 2 and 3 can run in parallel
        enable_progress=True,
        trace_name="Explicit Steps Chain",
    )

    # Display execution plan
    print(f"Total steps: {len(chain.steps)}")
    print(f"Execution plan: {chain.get_execution_plan()}")
    print("\nDependencies:")
    for step_num, deps in chain.get_step_dependencies().items():
        print(f"  Step {step_num}: depends on {deps if deps else 'nothing (can start immediately)'}")

    return chain


async def example_chain_builder():
    """
    Example 2: Using ChainBuilder for fluent chain construction + Execution.

    ChainBuilder provides a more readable way to construct chains,
    especially useful when building chains programmatically.
    """
    print("\n" + "=" * 60)
    print("Example 2: ChainBuilder Fluent API + Execution")
    print("=" * 60)

    chain = (
        ChainBuilder()
        .with_trace_name("ChainBuilder Chain")
        # First step: no dependencies
        .add_step(
            number=1,
            title="Data Overview",
            aim="Get a high-level overview of the data",
            reasoning_questions="What type of document is this? What is the main subject?",
            stage_action="Identify document type and key topics",
            example_reasoning="This is a quarterly financial report for Company XYZ",
        )
        # Second step: depends on first
        .add_step(
            number=2,
            title="Detailed Analysis",
            aim="Perform detailed analysis based on overview",
            reasoning_questions="What are the specific details worth noting?",
            stage_action="Deep dive into important sections",
            example_reasoning="Financial metrics show positive trends",
            dependencies=[1],
        )
        # Third step: depends on second
        .add_step(
            number=3,
            title="Conclusions",
            aim="Draw conclusions from the analysis",
            reasoning_questions="What conclusions can be drawn?",
            stage_action="Summarize findings and implications",
            example_reasoning="Company is performing well with some risks to monitor",
            dependencies=[2],
        )
        # Configure chain settings
        .with_max_workers(1)  # Sequential execution for this example
        .build()
    )

    print(f"Built chain with {len(chain.steps)} steps")
    print(f"Execution plan: {chain.get_execution_plan()}")

    return chain


async def example_legacy_api():
    """
    Example 3: Using the legacy StepDescription API + Execution.

    The legacy API is still supported for backward compatibility.
    It uses a single StepDescription class for all step types.
    """
    print("\n" + "=" * 60)
    print("Example 3: Legacy API (Backward Compatible) + Execution")
    print("=" * 60)

    # Legacy API uses StepDescription with step_type parameter
    steps = [
        StepDescription(
            number=1,
            title="Legacy Step",
            aim="Demonstrate legacy API with actual execution",
            reasoning_questions="How does the legacy API work?",
            stage_action="Show backward compatibility",
            example_reasoning="Legacy API works the same way with full execution support",
            # step_type defaults to StepType.LLM
        ),
    ]

    chain = ReasoningChain(steps=steps)
    print(f"Created chain with legacy API: {len(chain.steps)} steps")

    return chain


async def example_execution(chain: ReasoningChain, example_name: str, client=None):
    """
    Execute a chain with OpenRouter and display results.

    Args:
        chain: The reasoning chain to execute
        example_name: Name of the example for display
        client: OpenAI-compatible client to reuse (optional, will create if None)
    """
    print("\n" + "-" * 40)
    print(f"Executing {example_name}...")
    print("-" * 40)

    # Create context
    context = ReasoningContext(
        outer_context=SAMPLE_TEXT,
        api=client,
        model="unused",  # Not used with OpenAI-compatible clients
        language=Language.ENGLISH,
        system_prompt="You are a senior business analyst. Provide clear, actionable insights.",
    )

    # Execute chain
    result = await chain.execute_async(context)

    # Display results
    print(f"\nExecution: {format_status(result.success)}")
    print(f"Time: {result.total_execution_time:.2f}s")
    print(f"Steps completed: {len(result.get_successful_steps())}/{len(chain.steps)}")

    if result.success:
        print("\nFinal Output:")
        print("-" * 40)
        # Show last 500 chars of output for readability
        output = result.get_final_output()
        if len(output) > 500:
            print(output[:500] + "...")
        else:
            print(output)
        print("-" * 40)
    else:
        _print_failure_diagnostics(result, client)

    return result


def example_parallel_vs_sequential():
    """
    Example 5: Parallel vs Sequential Execution.

    Demonstrates how dependency structure affects execution order.
    """
    print("\n" + "=" * 60)
    print("Example 5: Parallel vs Sequential Execution")
    print("=" * 60)

    # Parallel chain: steps 1, 2, 3 can run simultaneously
    parallel_chain = (
        ChainBuilder()
        .add_step(
            number=1,
            title="Task A",
            aim="Independent task A",
            reasoning_questions="?",
            stage_action="Do A",
            example_reasoning="A done",
        )
        .add_step(
            number=2,
            title="Task B",
            aim="Independent task B",
            reasoning_questions="?",
            stage_action="Do B",
            example_reasoning="B done",
        )
        .add_step(
            number=3,
            title="Task C",
            aim="Independent task C",
            reasoning_questions="?",
            stage_action="Do C",
            example_reasoning="C done",
        )
        .add_step(
            number=4,
            title="Combine",
            aim="Combine results",
            reasoning_questions="?",
            stage_action="Combine",
            example_reasoning="Combined",
            dependencies=[1, 2, 3],
        )
        .with_max_workers(3)
        .build()
    )

    print("Parallel chain (max_workers=3):")
    print(f"  Execution plan: {parallel_chain.get_execution_plan()}")
    print("  Steps 1, 2, 3 run in parallel, then step 4")

    # Sequential chain: each step depends on previous
    sequential_chain = (
        ChainBuilder()
        .add_step(
            number=1,
            title="Step 1",
            aim="First step",
            reasoning_questions="?",
            stage_action="Do 1",
            example_reasoning="1 done",
        )
        .add_step(
            number=2,
            title="Step 2",
            aim="Second step",
            reasoning_questions="?",
            stage_action="Do 2",
            example_reasoning="2 done",
            dependencies=[1],
        )
        .add_step(
            number=3,
            title="Step 3",
            aim="Third step",
            reasoning_questions="?",
            stage_action="Do 3",
            example_reasoning="3 done",
            dependencies=[2],
        )
        .build()
    )

    print("\nSequential chain:")
    print(f"  Execution plan: {sequential_chain.get_execution_plan()}")
    print("  Each step waits for the previous one")

    return parallel_chain


def example_serialization():
    """
    Example 6: Chain Serialization.

    Demonstrates saving and loading chains to/from JSON.
    """
    print("\n" + "=" * 60)
    print("Example 6: Chain Serialization")
    print("=" * 60)

    chain = (
        ChainBuilder()
        .add_step(
            number=1,
            title="Serializable Step",
            aim="This step can be saved to JSON",
            reasoning_questions="How does serialization work?",
            stage_action="Save and load chains",
            example_reasoning="Chain serialization preserves all settings",
        )
        .build()
    )

    # Convert to dict
    chain_dict = chain.to_dict()
    print("Chain as dict:")
    print(f"  Keys: {list(chain_dict.keys())}")

    # Convert to JSON string
    json_str = chain.to_json()
    print("\nChain as JSON (first 200 chars):")
    print(f"  {json_str[:200]}...")

    # Restore from dict
    restored = ReasoningChain.from_dict(chain_dict)
    print(f"\nRestored from dict: {len(restored.steps)} steps")

    # Save to file
    # chain.save("my_chain.json")

    # Load from file
    # loaded = ReasoningChain.load("my_chain.json")

    print("\nSerialization methods:")
    print("  chain.to_dict() / ReasoningChain.from_dict(d)")
    print("  chain.to_json() / ReasoningChain.from_json(s)")
    print("  chain.save(path) / ReasoningChain.load(path)")

    return chain


async def main():
    """Run all basic examples with OpenRouter execution."""
    load_dotenv()
    print("CARL Basic Examples with OpenRouter")
    print("=" * 60)

    # Generate a shared session_id so all chains in this script
    # appear as one session in LangFuse.
    session_id = f"basic-example-{uuid.uuid4().hex[:8]}"
    print(f"LangFuse session_id: {session_id}")

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: OPENAI_API_KEY not set!")
        print("Chains will be displayed but not executed.")
        print("\nSet your API key:")
        print("  export OPENAI_API_KEY='sk-or-v1-...'")
        print("\nGet your key at: https://openrouter.ai/keys")
    else:
        print("\n✅ OPENAI_API_KEY is set")

    # Create client once and reuse it
    client = None
    if api_key:
        verify_ssl = _env_bool("OPENAI_SSL_VERIFY", default=True)
        client = create_openai_client(
            api_key=api_key,
            model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"),
            temperature=0.87,
            base_url=os.environ.get("OPENAI_BASE_URL"),
            verify_ssl=verify_ssl,
        )
        print(f"Provider: {client.config.base_url}")
        print(f"Model: {client.config.model}")
        print(f"SSL verify: {client.config.verify_ssl}")

    try:
        # Create and execute examples
        chain1 = await example_explicit_steps()
        chain1.session_id = session_id
        if client:
            await example_execution(chain1, "Explicit Steps Chain", client=client)

        chain2 = await example_chain_builder()
        chain2.session_id = session_id
        if client:
            await example_execution(chain2, "ChainBuilder Chain", client=client)

        chain3 = await example_legacy_api()
        chain3.session_id = session_id
        if client:
            await example_execution(chain3, "Legacy API Chain", client=client)

        # Non-execution examples
        chain4 = example_parallel_vs_sequential()
        chain4.session_id = session_id
        if client:
            await example_execution(chain4, "Parallel Execution Chain", client=client)

        example_serialization()

        # Flush any pending LangFuse events
        langfuse_flush()

        print("\n" + "=" * 60)
        print("Basic examples completed!")
        print("=" * 60)
        print("\nNext steps:")
        print("  - See openrouter_example.py for more OpenRouter features")
        print("  - See tool_steps_example.py for tool/memory integration")
        print("  - See structured_output_example.py for JSON schema validation")
    finally:
        # Properly close the client to avoid event loop issues
        if client and hasattr(client, "close"):
            try:
                await client.close()
            except Exception:
                pass  # Ignore cleanup errors


if __name__ == "__main__":
    asyncio.run(main())
