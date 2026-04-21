#!/usr/bin/env python3
"""
Example: Tool Steps and Memory Operations in CARL with OpenRouter.

This example demonstrates advanced CARL features:
- Tool steps for executing custom functions
- Memory steps for storing and retrieving data
- Transform steps for data transformations
- Combining LLM and non-LLM steps in chains

Prerequisites:
    pip install 'mmar-carl[openai]'
    # or: pip install openai

Environment variables:
    OPENAI_API_KEY: Your OpenRouter API key (https://openrouter.ai)

Usage:
    export OPENAI_API_KEY="sk-or-v1-..."
    python examples/tool_steps_example.py
"""

import asyncio
import json
import os
import uuid
from typing import Any

from mmar_carl import (
    ChainBuilder,
    Language,
    MemoryOperation,
    StepType,
    ToolStepConfig,
    ToolStepDescription,
    MemoryStepConfig,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    create_openai_client,
    create_step,
    LLMClientBase,
    langfuse_flush,
)
from examples.utils import format_status


# Mock client for tool-only chains (minimal interface)
class MockToolOnlyClient(LLMClientBase):
    """Minimal mock client for tool-only chains."""

    async def get_response(self, prompt: str) -> str:
        return "Tool execution complete - no LLM needed"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


# Sample data
FINANCIAL_DATA = {
    "revenue": [1000000, 1200000, 1100000, 1400000],
    "quarters": ["Q1", "Q2", "Q3", "Q4"],
    "year": 2024,
}


# ============================================================================
# Custom Tool Functions
# ============================================================================


def calculate_growth_rate(values: list[float]) -> dict[str, Any]:
    """Calculate quarter-over-quarter growth rates."""
    if len(values) < 2:
        return {"error": "Need at least 2 values"}

    growth_rates = []
    for i in range(1, len(values)):
        rate = ((values[i] - values[i - 1]) / values[i - 1]) * 100
        growth_rates.append(round(rate, 2))

    return {
        "growth_rates": growth_rates,
        "average_growth": round(sum(growth_rates) / len(growth_rates), 2),
        "max_growth": max(growth_rates),
        "min_growth": min(growth_rates),
    }


def calculate_statistics(values: list[float]) -> dict[str, Any]:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {"error": "Empty values list"}

    return {
        "count": len(values),
        "sum": sum(values),
        "mean": round(sum(values) / len(values), 2),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def format_currency(value: float, currency: str = "USD") -> str:
    """Format a number as currency."""
    symbols = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "RUB": "\u20bd"}
    symbol = symbols.get(currency, currency)
    return f"{symbol}{value:,.2f}"


def generate_report(
    title: str,
    metrics: dict[str, Any],
    growth: dict[str, Any],
) -> str:
    """Generate a formatted report from metrics and growth data."""
    report = f"""
{"=" * 50}
{title}
{"=" * 50}

Key Metrics:
- Total: {format_currency(metrics.get("sum", 0))}
- Average: {format_currency(metrics.get("mean", 0))}
- Range: {format_currency(metrics.get("range", 0))}

Growth Analysis:
- Average Growth: {growth.get("average_growth", 0)}%
- Best Quarter: +{growth.get("max_growth", 0)}%
- Worst Quarter: {growth.get("min_growth", 0)}%

{"=" * 50}
"""
    return report


# ============================================================================
# Examples
# ============================================================================


def get_revenue_data() -> list[float]:
    """Get revenue data from financial data."""
    return FINANCIAL_DATA["revenue"]


async def example_tool_steps(session_id: str | None = None):
    """
    Example 1: Using Tool Steps.

    Tool steps execute registered Python functions and pass results
    to subsequent steps.
    """
    print("\n" + "=" * 60)
    print("Example 1: Tool Steps")
    print("=" * 60)

    # Create chain with tool steps
    builder = ChainBuilder()
    if session_id:
        builder = builder.with_session_id(session_id)
    chain = (
        builder.with_trace_name("Tool Steps Example")
        # Tool step: Get revenue data
        .add_tool_step(
            number=1,
            title="Get Revenue Data",
            tool_name="get_revenue_data",
            input_mapping={},
        )
        # Tool step: Calculate statistics
        .add_tool_step(
            number=2,
            title="Calculate Statistics",
            tool_name="calculate_statistics",
            input_mapping={"values": "$metadata.step_1"},
            dependencies=[1],
        )
        # Tool step: Calculate growth
        .add_tool_step(
            number=3,
            title="Calculate Growth",
            tool_name="calculate_growth_rate",
            input_mapping={"values": "$metadata.step_1"},
            dependencies=[1],
        )
        # Tool step: Generate report
        .add_tool_step(
            number=4,
            title="Generate Report",
            tool_name="generate_report",
            input_mapping={
                "title": "Revenue Analysis 2024",
                "metrics": "$metadata.step_2",  # Result from step 2
                "growth": "$metadata.step_3",  # Result from step 3
            },
            dependencies=[2, 3],
        )
        .with_max_workers(2)  # Steps 2 and 3 can run in parallel
        .build()
    )

    print(f"Chain has {len(chain.steps)} tool steps")
    print(f"Execution plan: {chain.get_execution_plan()}")

    # Create context (tool-only chains don't need LLM client)
    context = ReasoningContext(
        outer_context="",  # Not used when data comes from tools
        api=MockToolOnlyClient(),
        model="unused",
    )

    # Register tools
    context.register_tool("get_revenue_data", get_revenue_data)
    context.register_tool("calculate_statistics", calculate_statistics)
    context.register_tool("calculate_growth_rate", calculate_growth_rate)
    context.register_tool("generate_report", generate_report)

    # Execute
    result = await chain.execute_async(context)

    print(f"\nExecution: {format_status(result.success)}")
    if result.success:
        print("\nGenerated Report:")
        print(result.get_final_output())
    else:
        print("\nErrors:")
        for step_result in result.step_results:
            if not step_result.success:
                print(f"  Step {step_result.step_number}: {step_result.error_message}")

    return chain


async def example_memory_steps(session_id: str | None = None):
    """
    Example 2: Using Memory Steps.

    Memory steps store and retrieve data between steps using
    a key-value store with optional namespaces.
    """
    print("\n" + "=" * 60)
    print("Example 2: Memory Steps")
    print("=" * 60)

    def get_revenue_data() -> list[float]:
        """Get revenue data from financial data."""
        return FINANCIAL_DATA["revenue"]

    def get_year() -> int:
        """Get year from financial data."""
        return FINANCIAL_DATA["year"]

    builder = ChainBuilder()
    if session_id:
        builder = builder.with_session_id(session_id)
    chain = (
        builder.with_trace_name("Memory Steps Example")
        # Tool step: Get revenue data
        .add_tool_step(
            number=1,
            title="Get Revenue Data",
            tool_name="get_revenue_data",
            input_mapping={},
        )
        # Tool step: Calculate something
        .add_tool_step(
            number=2,
            title="Calculate Metrics",
            tool_name="calculate_statistics",
            input_mapping={"values": "$metadata.step_1"},
            dependencies=[1],
        )
        # Memory step: Store the result
        .add_memory_step(
            number=3,
            title="Store Metrics",
            operation="write",
            memory_key="revenue_metrics",
            value_source="$metadata.step_2",  # Store result from step 2
            namespace="analysis",
            dependencies=[2],
        )
        # Memory step: Store additional data
        .add_memory_step(
            number=4,
            title="Store Year",
            operation="write",
            memory_key="report_year",
            value_source="$outer_context",
            namespace="analysis",
        )
        # Memory step: Read stored data (for demonstration)
        .add_memory_step(
            number=5,
            title="Read Metrics",
            operation="read",
            memory_key="revenue_metrics",
            namespace="analysis",
            dependencies=[3],
        )
        # Memory step: List all keys in namespace
        .add_memory_step(
            number=6,
            title="List Analysis Keys",
            operation="list",
            memory_key="",  # Not used for list operation
            namespace="analysis",
            dependencies=[3, 4],
        )
        .build()
    )

    print(f"Chain has {len(chain.steps)} memory/tool steps")
    print("\nMemory operations available:")
    print("  - write: Store a value")
    print("  - read: Retrieve a value")
    print("  - append: Append to a list")
    print("  - delete: Remove a key")
    print("  - list: List all keys in namespace")

    # Create context and execute
    context = ReasoningContext(
        outer_context=str(FINANCIAL_DATA.get("year", 2024)),  # Store year as string for memory step
        api=MockToolOnlyClient(),
        model="unused",
    )
    context.register_tool("get_revenue_data", get_revenue_data)
    context.register_tool("calculate_statistics", calculate_statistics)
    context.register_tool("get_year", get_year)

    result = await chain.execute_async(context)

    print(f"\nExecution: {format_status(result.success)}")
    if result.success:
        # Show memory list result
        memory_list = result.step_results[-1].result
        print(f"\nKeys in 'analysis' namespace: {memory_list}")
    else:
        print("\nErrors:")
        for step_result in result.step_results:
            if not step_result.success:
                print(f"  Step {step_result.step_number}: {step_result.error_message}")

    return chain


async def example_transform_steps(session_id: str | None = None):
    """
    Example 3: Using Transform Steps.

    Transform steps perform data transformations without calling an LLM.
    Useful for data extraction, formatting, and manipulation.
    """
    print("\n" + "=" * 60)
    print("Example 3: Transform Steps")
    print("=" * 60)

    def get_financial_data() -> dict:
        """Get financial data."""
        return FINANCIAL_DATA

    builder = ChainBuilder()
    if session_id:
        builder = builder.with_session_id(session_id)
    chain = (
        builder.with_trace_name("Transform Steps Example")
        # Tool step: Get data as dict
        .add_tool_step(
            number=1,
            title="Get Financial Data",
            tool_name="get_financial_data",
            input_mapping={},
        )
        # Transform: Extract revenue array
        .add_transform_step(
            number=2,
            title="Extract Revenue",
            transform_type="extract",
            input_key="history[-1]",
            expression="revenue",
            dependencies=[1],
        )
        # Transform: Format as JSON
        .add_transform_step(
            number=3,
            title="Format as JSON",
            transform_type="format",
            input_key="history[-1]",
            expression="json",
            dependencies=[2],
        )
        .build()
    )

    print(f"Chain has {len(chain.steps)} transform steps")
    print("\nTransform types:")
    print("  - extract: Extract a field from data")
    print("  - format: Format data (json, csv, etc.)")
    print("  - map: Apply transformation to each item")
    print("  - filter: Filter items based on condition")

    # Create context and execute
    context = ReasoningContext(
        outer_context="",
        api=MockToolOnlyClient(),
        model="unused",
    )
    context.register_tool("get_financial_data", get_financial_data)

    result = await chain.execute_async(context)

    print(f"\nExecution: {format_status(result.success)}")
    if result.success:
        print("\nTransformed Revenue (JSON):")
        print(result.get_final_output())

    return chain


async def example_mixed_chain(session_id: str | None = None):
    """
    Example 4: Mixed Chain with LLM, Tool, and Memory Steps.

    This demonstrates a realistic workflow combining different step types.
    """
    print("\n" + "=" * 60)
    print("Example 4: Mixed Chain (LLM + Tool + Memory)")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Skipping: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='sk-or-v1-...'")
        return None

    # Wrapper tool to get revenue data
    def get_revenue_array() -> list[float]:
        """Get revenue data as list of floats."""
        return FINANCIAL_DATA["revenue"]

    def format_metrics_for_llm(metrics: dict, growth: dict) -> str:
        """Format metrics and growth data for LLM input."""
        return f"""
Financial Data Analysis:

Statistics:
- Total Revenue: ${metrics.get("sum", 0):,.0f}
- Average: ${metrics.get("mean", 0):,.0f}
- Min: ${metrics.get("min", 0):,.0f}
- Max: ${metrics.get("max", 0):,.0f}

Growth Rates:
{json.dumps(growth.get("growth_rates", []), indent=2)}

Average Growth: {growth.get("average_growth", 0)}%
Best Quarter: +{growth.get("max_growth", 0)}%
Worst Quarter: {growth.get("min_growth", 0)}%
"""

    # Using typed step descriptions for clarity
    steps = [
        # Step 1: Tool - Get revenue data
        ToolStepDescription(
            number=1,
            title="Get Revenue Data",
            config=ToolStepConfig(
                tool_name="get_revenue_array",
                input_mapping={},
            ),
        ),
        # Step 2: Tool - Calculate statistics
        ToolStepDescription(
            number=2,
            title="Calculate Statistics",
            config=ToolStepConfig(
                tool_name="calculate_statistics",
                input_mapping={"values": "$metadata.step_1"},
            ),
            dependencies=[1],
        ),
        # Step 3: Tool - Calculate growth
        ToolStepDescription(
            number=3,
            title="Calculate Growth",
            config=ToolStepConfig(
                tool_name="calculate_growth_rate",
                input_mapping={"values": "$metadata.step_1"},
            ),
            dependencies=[1],
        ),
        # Step 4: Tool - Format for LLM
        ToolStepDescription(
            number=4,
            title="Format Analysis for LLM",
            config=ToolStepConfig(
                tool_name="format_metrics_for_llm",
                input_mapping={
                    "metrics": "$metadata.step_2",
                    "growth": "$metadata.step_3",
                },
            ),
            dependencies=[2, 3],
        ),
        # Step 5: LLM - Analyze results
        LLMStepDescription(
            number=5,
            title="LLM Analysis",
            aim="Interpret the calculated metrics and provide insights",
            reasoning_questions="What do the statistics and growth rates indicate about business health?",
            stage_action="Analyze the numbers and provide business interpretation",
            example_reasoning="Strong average growth of X% suggests positive momentum",
            step_context_queries=["revenue", "growth"],
            dependencies=[4],
        ),
        # Step 6: LLM - Generate recommendations
        LLMStepDescription(
            number=6,
            title="Strategic Recommendations",
            aim="Provide actionable recommendations based on analysis",
            reasoning_questions="What actions should be taken based on these findings?",
            stage_action="Generate specific, actionable recommendations",
            example_reasoning="Given the growth trend, recommend increasing investment in Q2",
            dependencies=[5],
        ),
    ]

    chain = ReasoningChain(
        steps=steps,
        max_workers=2,
        trace_name="Mixed Chain Example",
        session_id=session_id,
    )

    print(f"Chain has {len(chain.steps)} mixed steps:")
    for step in chain.steps:
        step_type = step.step_type.value if hasattr(step, "step_type") else "llm"
        print(f"  Step {step.number}: {step.title} ({step_type})")

    print(f"\nExecution plan: {chain.get_execution_plan()}")

    # Create OpenRouter client
    client = create_openai_client(
        api_key=api_key,
        model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"),
        temperature=0.7,
    )

    # Create context - pass data as JSON string
    context = ReasoningContext(
        outer_context=json.dumps(FINANCIAL_DATA, indent=2),
        api=client,
        model="unused",
        language=Language.ENGLISH,
        system_prompt="You are a senior financial analyst. Provide clear, actionable insights.",
    )

    # Register tools
    context.register_tool("get_revenue_array", get_revenue_array)
    context.register_tool("calculate_statistics", calculate_statistics)
    context.register_tool("calculate_growth_rate", calculate_growth_rate)
    context.register_tool("format_metrics_for_llm", format_metrics_for_llm)

    # Execute
    result = await chain.execute_async(context)

    print(f"\nExecution: {format_status(result.success)}")
    print(f"Time: {result.total_execution_time:.2f}s")

    if result.success:
        print("\nFinal Recommendations:")
        print("-" * 40)
        output = result.get_final_output()
        if len(output) > 500:
            print(output[:500] + "...")
        else:
            print(output)
        print("-" * 40)
    else:
        print("\nErrors:")
        for step_result in result.step_results:
            if not step_result.success:
                print(f"  Step {step_result.step_number}: {step_result.error_message}")

    return chain


def example_create_step_factory():
    """
    Example 5: Using create_step() Factory Function.

    The create_step() factory provides a unified way to create
    any type of step.
    """
    print("\n" + "=" * 60)
    print("Example 5: create_step() Factory Function")
    print("=" * 60)

    # Create different step types using the factory
    llm_step = create_step(
        number=1,
        title="LLM Step",
        step_type=StepType.LLM,
        aim="Analyze data",
        reasoning_questions="What patterns exist?",
        stage_action="Find patterns",
        example_reasoning="Patterns found",
    )

    tool_step = create_step(
        number=2,
        title="Tool Step",
        step_type=StepType.TOOL,
        config=ToolStepConfig(
            tool_name="my_tool",
            input_mapping={"x": "$metadata.step_1"},
        ),
        dependencies=[1],
    )

    memory_step = create_step(
        number=3,
        title="Memory Step",
        step_type=StepType.MEMORY,
        config=MemoryStepConfig(
            operation=MemoryOperation.WRITE,
            memory_key="result",
            value_source="$metadata.step_2",
        ),
        dependencies=[2],
    )

    print("Created steps using factory:")
    print(f"  1. {llm_step.title} (type: {llm_step.step_type})")
    print(f"  2. {tool_step.title} (type: {tool_step.step_type})")
    print(f"  3. {memory_step.title} (type: {memory_step.step_type})")


async def main():
    """Run all tool/memory examples with OpenRouter execution."""
    print("CARL Tool & Memory Examples with OpenRouter")
    print("=" * 60)

    # Generate a shared session_id so all chains in this script
    # appear as one session in LangFuse.
    session_id = f"tool-steps-example-{uuid.uuid4().hex[:8]}"
    print(f"LangFuse session_id: {session_id}")

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set!")
        print("Tool-only examples will run, but LLM examples will be skipped.")
        print("\nSet your API key:")
        print("  export OPENAI_API_KEY='sk-or-v1-...'")
        print("\nGet your key at: https://openrouter.ai/keys")
    else:
        print("\n✅ OPENAI_API_KEY is set")

    # Tool-only examples (no API key needed)
    await example_tool_steps(session_id=session_id)
    await example_memory_steps(session_id=session_id)
    await example_transform_steps(session_id=session_id)
    example_create_step_factory()

    # Mixed example requires LLM
    await example_mixed_chain(session_id=session_id)

    # Flush any pending LangFuse events
    langfuse_flush()

    print("\n" + "=" * 60)
    print("Tool & Memory examples completed!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  - Tool steps execute Python functions")
    print("  - Memory steps persist data between steps")
    print("  - Transform steps manipulate data without LLM")
    print("  - Mix step types for powerful workflows")
    print("  - Use input_mapping to reference previous results")
    print("  - Use trace_name and session_id for LangFuse observability")
    print("\nNext steps:")
    print("  - See basic_chain_example.py for chain fundamentals")
    print("  - See openrouter_example.py for more OpenRouter features")
    print("  - See structured_output_example.py for JSON schema validation")


if __name__ == "__main__":
    asyncio.run(main())
