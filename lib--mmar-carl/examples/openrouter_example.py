#!/usr/bin/env python3
"""
Example: Running CARL chains with OpenRouter.

This example demonstrates how to use CARL with OpenRouter or any other
OpenAI-compatible API (Azure OpenAI, local LLMs like Ollama, vLLM, etc.).

Prerequisites:
    pip install 'mmar-carl[openai]'
    # or: pip install openai

Environment variables:
    OPENAI_API_KEY: Your OpenAI-compatible API key (OpenRouter: https://openrouter.ai)

Usage:
    export OPENAI_API_KEY="sk-or-v1-..."
    python examples/openrouter_example.py
"""

import asyncio
import os

from mmar_carl import (
    ChainBuilder,
    Language,
    LLMStepDescription,
    OpenAIClientConfig,
    OpenAICompatibleClient,
    ReasoningChain,
    ReasoningContext,
    create_openai_client,
    langfuse_flush,
)
from examples.utils import format_status


# Sample financial data for analysis
SAMPLE_DATA = """Period,Revenue,Expenses,Profit,Growth_Rate
2024-Q1,1500000,1200000,300000,5.2%
2024-Q2,1650000,1280000,370000,10.0%
2024-Q3,1720000,1350000,370000,4.2%
2024-Q4,1850000,1400000,450000,7.6%"""


def create_analysis_chain() -> ReasoningChain:
    """Create a financial analysis reasoning chain."""
    steps = [
        LLMStepDescription(
            number=1,
            title="Revenue Analysis",
            aim="Analyze revenue trends and growth patterns",
            reasoning_questions="What is the revenue trend? Is growth accelerating or decelerating?",
            stage_action="Calculate quarter-over-quarter revenue growth and identify patterns",
            example_reasoning="Revenue shows consistent growth with Q2 having the highest growth rate at 10%",
            step_context_queries=["Revenue", "Growth_Rate"],
        ),
        LLMStepDescription(
            number=2,
            title="Profitability Analysis",
            aim="Evaluate profit margins and expense efficiency",
            reasoning_questions="How are profit margins evolving? Are expenses being managed effectively?",
            stage_action="Calculate profit margin = Profit/Revenue and analyze expense ratio",
            example_reasoning="Profit margin improved from 20% in Q1 to 24.3% in Q4, indicating better expense management",
            step_context_queries=["Revenue", "Expenses", "Profit"],
            dependencies=[1],
        ),
        LLMStepDescription(
            number=3,
            title="Summary and Recommendations",
            aim="Provide actionable insights and recommendations",
            reasoning_questions="What are the key takeaways? What actions should be taken?",
            stage_action="Synthesize findings and provide strategic recommendations",
            example_reasoning="Strong financial performance with improving margins suggests opportunity for expansion",
            dependencies=[1, 2],
        ),
    ]
    return ReasoningChain(steps=steps, max_workers=2, trace_name="Financial Analysis")


async def example_with_openrouter():
    """
    Example 1: Using OpenRouter with explicit configuration.

    This shows how to configure OpenAICompatibleClient with full control
    over all parameters including custom headers for OpenRouter.
    """
    print("\n" + "=" * 60)
    print("Example 1: OpenRouter with Explicit Configuration")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Skipping: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='sk-or-v1-...'")
        return

    # Create OpenRouter client with full configuration
    config = OpenAIClientConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=os.environ.get(
            "DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"
        ),  # Or use other models like "openai/gpt-4o"
        temperature=0.7,
        max_tokens=2048,
        timeout=120.0,
        extra_headers={
            "HTTP-Referer": "https://github.com/your-org/your-project",
            "X-Title": "CARL Financial Analysis",
        },
    )
    client = OpenAICompatibleClient(config)

    # Create context with the OpenRouter client
    context = ReasoningContext(
        outer_context=SAMPLE_DATA,
        api=client,
        model="unused",  # Not used with OpenAI-compatible clients
        language=Language.ENGLISH,
        system_prompt="You are a senior financial analyst. Provide data-driven insights.",
    )

    # Create and execute chain
    chain = create_analysis_chain()
    print(f"Executing chain with {len(chain.steps)} steps...")
    print(f"Model: {config.model}")

    result = await chain.execute_async(context)

    print(f"\nExecution: {format_status(result.success)}")
    print(f"Time: {result.total_execution_time:.2f}s")

    if result.success:
        print("\nAnalysis Result:")
        print("-" * 40)
        print(result.get_final_output())


async def example_with_factory_function():
    """
    Example 2: Using the create_openai_client() factory function.

    This is a simpler way to create an OpenRouter client when you don't
    need all the configuration options.
    """
    print("\n" + "=" * 60)
    print("Example 2: Using Factory Function")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Skipping: OPENAI_API_KEY not set")
        return

    # Simple factory function usage
    client = create_openai_client(
        api_key=api_key,
        model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"),
        temperature=0.5,
    )

    context = ReasoningContext(
        outer_context=SAMPLE_DATA,
        api=client,
        model="unused",
        language=Language.ENGLISH,
        system_prompt="You are a concise financial analyst. Be brief but insightful.",
    )

    # Use ChainBuilder for fluent chain construction
    chain = (
        ChainBuilder()
        .add_step(
            number=1,
            title="Quick Analysis",
            aim="Provide a quick financial overview",
            reasoning_questions="What are the key financial highlights?",
            stage_action="Summarize key metrics and trends",
            example_reasoning="Revenue grew 23% over the year with improving margins",
            step_context_queries=["Revenue", "Profit", "Growth_Rate"],
        )
        .with_max_workers(1)
        .with_trace_name("Factory Function Chain")
        .build()
    )

    print(f"Model: {client.config.model}")
    result = await chain.execute_async(context)

    print(f"\nExecution: {format_status(result.success)}")
    if result.success:
        print("\nQuick Analysis:")
        print(result.get_final_output())


async def example_with_local_llm():
    """
    Example 3: Using a local LLM with OpenAI-compatible API.

    This shows how to use CARL with local LLMs like Ollama, LM Studio,
    vLLM, or any other service that provides an OpenAI-compatible API.
    """
    print("\n" + "=" * 60)
    print("Example 3: Local LLM (Ollama/LM Studio)")
    print("=" * 60)

    # Check if local LLM is available
    local_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")
    local_model = os.environ.get("LOCAL_LLM_MODEL", "llama3")

    print(f"Local LLM URL: {local_url}")
    print(f"Local Model: {local_model}")
    print("\nNote: This example requires a running local LLM server.")
    print("For Ollama: ollama serve && ollama pull llama3")
    print("For LM Studio: Start the server in LM Studio settings")

    # Create client for local LLM
    client = create_openai_client(
        api_key="not-needed",  # Most local LLMs don't require API keys
        model=local_model,
        base_url=local_url,
        temperature=0.7,
        timeout=300.0,  # Local models may be slower
    )

    context = ReasoningContext(
        outer_context=SAMPLE_DATA,
        api=client,
        model="unused",
        language=Language.ENGLISH,
        system_prompt="You are a helpful financial analyst.",
    )

    chain = (
        ChainBuilder()
        .add_step(
            number=1,
            title="Local Analysis",
            aim="Analyze data using local LLM",
            reasoning_questions="What insights can you provide from this data?",
            stage_action="Analyze the financial metrics",
            example_reasoning="The data shows positive trends",
        )
        .with_trace_name("Local LLM Analysis")
        .build()
    )

    try:
        result = await chain.execute_async(context)
        print(f"\nExecution: {format_status(result.success)}")
        if result.success:
            print("\nLocal LLM Analysis:")
            print(result.get_final_output())
    except Exception as e:
        print(f"\nCould not connect to local LLM: {e}")
        print("Make sure your local LLM server is running.")


async def example_different_models():
    """
    Example 4: Using different models for different steps.

    This demonstrates how you can use different OpenRouter models
    for different reasoning chains based on complexity and cost.
    """
    print("\n" + "=" * 60)
    print("Example 4: Different Models for Different Tasks")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Skipping: OPENAI_API_KEY not set")
        return

    # Fast model for quick analysis
    fast_client = create_openai_client(
        api_key=api_key,
        model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"),  # Fast and cheap
        temperature=0.3,
    )

    # Powerful model for complex reasoning
    powerful_client = create_openai_client(
        api_key=api_key,
        model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"),
        temperature=0.7,
    )

    # Simple task with fast model
    simple_context = ReasoningContext(
        outer_context=SAMPLE_DATA,
        api=fast_client,
        model="unused",
        language=Language.ENGLISH,
        system_prompt="Extract key numbers from the data.",
    )

    simple_chain = (
        ChainBuilder()
        .add_step(
            number=1,
            title="Extract Metrics",
            aim="Extract key financial metrics",
            reasoning_questions="What are the key numbers?",
            stage_action="List the main metrics",
            example_reasoning="Q4 Revenue: $1.85M, Profit: $450K",
        )
        .with_trace_name("Simple Metrics Extraction")
        .build()
    )

    print("Running simple extraction with GPT-4o-mini...")
    simple_result = simple_chain.execute(simple_context)
    if simple_result.success:
        print(f"Fast model result:\n{simple_result.get_final_output()}")

    # Complex task with powerful model
    complex_context = ReasoningContext(
        outer_context=SAMPLE_DATA,
        api=powerful_client,
        model="unused",
        language=Language.ENGLISH,
        system_prompt="You are a senior financial strategist. Provide deep analysis.",
    )

    complex_chain = create_analysis_chain()

    print("\nRunning complex analysis with Claude 3.5 Sonnet...")
    complex_result = await complex_chain.execute_async(complex_context)
    if complex_result.success:
        print(f"Powerful model result:\n{complex_result.get_final_output()}")


async def main():
    """Run all examples."""
    print("CARL + OpenRouter Examples")
    print("=" * 60)
    print("\nThese examples demonstrate using CARL with OpenAI-compatible APIs.")
    print("Set OPENAI_API_KEY to run OpenRouter examples.")
    print("Start a local LLM server to run local LLM examples.")

    await example_with_openrouter()
    await example_with_factory_function()
    await example_with_local_llm()
    await example_different_models()

    # Flush any pending LangFuse events
    langfuse_flush()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
