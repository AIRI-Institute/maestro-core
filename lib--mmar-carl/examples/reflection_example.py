#!/usr/bin/env python3
"""
Example: CARL Reflection Feature

This example demonstrates the reflection functionality for analyzing
chain execution results after completion.

Prerequisites:
    pip install 'mmar-carl[openai]'
    # or: pip install openai

Environment variables:
    OPENAI_API_KEY: Your OpenRouter API key (get one at https://openrouter.ai)

Usage:
    export OPENAI_API_KEY="sk-or-v1-..."
    python examples/reflection_example.py
"""

import asyncio
import os

from mmar_carl import (
    ChainBuilder,
    Language,
    LLMStepDescription,
    create_openai_client,
    ReasoningChain,
    ReasoningContext,
    ReflectionOptions,
)
from examples.utils import format_status


# Sample customer feedback data
SAMPLE_FEEDBACK = """CustomerID,Product,Rating,Comment,Date
C001,PremiumWidget,5,"Absolutely love it! Best purchase I've made this year.",2024-11-15
C002,PremiumWidget,4,"Great product but shipping was slow.",2024-11-14
C003,StandardGadget,3,"It's okay for the price. Does what it's supposed to do.",2024-11-13
C004,PremiumWidget,5,"Exceptional quality! Would definitely recommend to others.",2024-11-12
C005,StandardGadget,2,"Stopped working after a week. Disappointed.",2024-11-11
C006,PremiumWidget,5,"Worth every penny! The premium quality really shows.",2024-11-10
C007,StandardGadget,4,"Good value for money. Minor issues but overall satisfied.",2024-11-09
C008,PremiumWidget,1,"Terrible experience. Product arrived damaged.",2024-11-08
C009,StandardGadget,3,"Average product. Nothing special but nothing terrible either.",2024-11-07
C010,PremiumWidget,5,"Incredible! Exceeded all my expectations. Will buy again!",2024-11-06"""


def create_sentiment_chain() -> ReasoningChain:
    """Create a sentiment analysis reasoning chain."""
    steps = [
        LLMStepDescription(
            number=1,
            title="Extract Key Themes",
            aim="Identify the main themes and topics in customer feedback",
            reasoning_questions="What are customers mentioning most? What themes emerge?",
            stage_action="Extract and categorize themes from comments (quality, shipping, price, etc.)",
            example_reasoning="Customers frequently mention product quality, shipping speed, and value for money",
            step_context_queries=["Comment"],
        ),
        LLMStepDescription(
            number=2,
            title="Analyze Sentiment Patterns",
            aim="Analyze sentiment patterns across different products and ratings",
            reasoning_questions="How does sentiment differ between products? What correlates with ratings?",
            stage_action="Correlate themes with ratings and identify patterns",
            example_reasoning="PremiumWidget has higher ratings but some quality concerns; StandardGadget shows mixed sentiment",
            step_context_queries=["Product", "Rating", "Comment"],
            dependencies=[1],
        ),
        LLMStepDescription(
            number=3,
            title="Generate Actionable Insights",
            aim="Provide actionable recommendations based on customer feedback",
            reasoning_questions="What should the company improve? What are they doing well?",
            stage_action="Synthesize findings into specific recommendations",
            example_reasoning="Focus on improving quality control for PremiumWidget and enhancing StandardGadget features",
            dependencies=[1, 2],
        ),
    ]
    return ReasoningChain(steps=steps, max_workers=2, trace_name="Sentiment Analysis")


def create_simple_chain() -> ReasoningChain:
    """Create a simple chain for basic reflection demonstration."""
    steps = [
        LLMStepDescription(
            number=1,
            title="Quick Summary",
            aim="Provide a quick summary of the feedback",
            reasoning_questions="What are the key takeaways?",
            stage_action="Summarize the main points",
            example_reasoning="Most customers are satisfied with PremiumWidget, though some quality issues exist",
            step_context_queries=["Comment", "Rating"],
        ),
    ]
    return ReasoningChain(steps=steps, max_workers=1, trace_name="Quick Summary")


def create_chain_with_potential_failure() -> ReasoningChain:
    """Create a chain that demonstrates reflection with partial failures."""
    steps = [
        LLMStepDescription(
            number=1,
            title="Analyze Feedback",
            aim="Analyze customer feedback in detail",
            reasoning_questions="What patterns exist in the feedback?",
            stage_action="Extract insights from comments",
            example_reasoning="Feedback shows general satisfaction with some issues",
            step_context_queries=["Comment"],
        ),
        LLMStepDescription(
            number=2,
            title="Competitive Analysis",
            aim="Compare against competitor products (simulated failure)",
            reasoning_questions="How do we compare to competitors?",
            stage_action="Analyze competitive positioning",
            example_reasoning="This step will fail due to missing competitor data",
            step_context_queries=["CompetitorData"],  # This doesn't exist - may cause issues
            dependencies=[1],
        ),
        LLMStepDescription(
            number=3,
            title="Final Recommendations",
            aim="Provide final recommendations despite incomplete analysis",
            reasoning_questions="What can we recommend based on available data?",
            stage_action="Generate recommendations from available information",
            example_reasoning="Focus on improving quality control and shipping times",
            dependencies=[1],
        ),
    ]
    return ReasoningChain(steps=steps, max_workers=2, trace_name="Analysis with Potential Failures")


async def execute_and_reflect(
    chain: ReasoningChain,
    context: ReasoningContext,
    task_description: str,
    example_name: str,
) -> None:
    """
    Execute a chain and generate reflection.

    Args:
        chain: The reasoning chain to execute
        context: The reasoning context
        task_description: Description of the task for reflection
        example_name: Name of the example for display
    """
    print("\n" + "=" * 70)
    print(f"{example_name}")
    print("=" * 70)

    # Execute the chain
    _ = context  # Context is used by chain.execute()
    result = await chain.execute_async(context)

    # Display execution stats
    print(f"\nExecution Status: {format_status(result.success)}")
    print(f"Total Steps: {len(result.step_results)}")
    print(f"Successful Steps: {len(result.get_successful_steps())}")
    print(f"Failed Steps: {len(result.get_failed_steps())}")
    if result.total_execution_time:
        print(f"Execution Time: {result.total_execution_time:.2f}s")

    # Generate reflection
    print("\n" + "-" * 70)
    print("Generating Reflection...")
    print("-" * 70)

    try:
        reflection = chain.reflect(task_description=task_description)
        print("\nReflection:")
        print(reflection)
    except RuntimeError as e:
        print(f"\nReflection Error: {e}")
    print()


async def example_basic_reflection():
    """Example 1: Basic synchronous reflection."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nSkipping: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='sk-or-v1-...'")
        return

    client = create_openai_client(
        api_key=api_key,
        model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"),
        temperature=0.5,
    )

    context = ReasoningContext(
        outer_context=SAMPLE_FEEDBACK,
        api=client,
        model="unused",
        language=Language.ENGLISH,
        system_prompt="You are a customer experience analyst specializing in feedback analysis.",
    )

    chain = create_sentiment_chain()

    await execute_and_reflect(
        chain=chain,
        context=context,
        task_description="Analyze customer feedback to identify key themes, sentiment patterns, and provide actionable recommendations for improving customer satisfaction.",
        example_name="Example 1: Basic Reflection (Sentiment Analysis)",
    )


async def example_async_reflection():
    """Example 2: Async reflection using reflect_async()."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nSkipping: OPENAI_API_KEY not set")
        return

    client = create_openai_client(
        api_key=api_key,
        model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"),
        temperature=0.5,
    )

    context = ReasoningContext(
        outer_context=SAMPLE_FEEDBACK,
        api=client,
        model="unused",
        language=Language.ENGLISH,
        system_prompt="You are a customer experience analyst.",
    )

    chain = create_simple_chain()

    print("\n" + "=" * 70)
    print("Example 2: Async Reflection (Quick Summary)")
    print("=" * 70)

    # Execute async
    result = await chain.execute_async(context)

    print(f"\nExecution Status: {format_status(result.success)}")
    print(f"Execution Time: {result.total_execution_time:.2f}s")

    # Reflect async
    print("\n" + "-" * 70)
    print("Generating Async Reflection...")
    print("-" * 70)

    reflection = await chain.reflect_async(task_description="Provide a quick summary of customer feedback highlights.")
    print("\nReflection:")
    print(reflection)
    print()


async def example_reflection_with_failures():
    """Example 3: Reflection with failed steps."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nSkipping: OPENAI_API_KEY not set")
        return

    client = create_openai_client(
        api_key=api_key,
        model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"),
        temperature=0.5,
    )

    context = ReasoningContext(
        outer_context=SAMPLE_FEEDBACK,
        api=client,
        model="unused",
        language=Language.ENGLISH,
        system_prompt="You are a customer experience analyst.",
    )

    chain = create_chain_with_potential_failure()

    await execute_and_reflect(
        chain=chain,
        context=context,
        task_description="Analyze customer feedback and provide competitive insights. Note that competitor data may not be available.",
        example_name="Example 3: Reflection with Potential Failures",
    )


async def example_russian_reflection():
    """Example 4: Russian language reflection."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nSkipping: OPENAI_API_KEY not set")
        return

    client = create_openai_client(
        api_key=api_key,
        model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"),
        temperature=0.5,
    )

    context = ReasoningContext(
        outer_context=SAMPLE_FEEDBACK,
        api=client,
        model="unused",
        language=Language.RUSSIAN,
        system_prompt="Вы — аналитик клиентского опыта, специализирующийся на анализе отзывов.",
    )

    chain = (
        ChainBuilder()
        .add_step(
            number=1,
            title="Анализ отзывов",
            aim="Проанализировать отзывы клиентов",
            reasoning_questions="Какие основные темы в отзывах?",
            stage_action="Извлечь ключевые инсайты из комментариев",
            example_reasoning="Клиенты в целом довольны качеством, но отмечают проблемы с доставкой",
            step_context_queries=["Comment", "Rating"],
        )
        .add_step(
            number=2,
            title="Рекомендации",
            aim="Предоставить рекомендации по улучшению",
            reasoning_questions="Что можно улучшить?",
            stage_action="Синтезировать выводы в конкретные рекомендации",
            example_reasoning="Улучшить контроль качества и скорость доставки",
            dependencies=[1],
        )
        .with_trace_name("Russian Language Reflection")
        .build()
    )

    await execute_and_reflect(
        chain=chain,
        context=context,
        task_description="Проанализировать отзывы клиентов, выявить основные темы и предоставить рекомендации по улучшению удовлетворенности клиентов.",
        example_name="Example 4: Russian Language Reflection (Анализ на русском языке)",
    )


async def example_reflection_before_execution():
    """Example 5: Demonstrating RuntimeError when reflecting before execution."""
    print("\n" + "=" * 70)
    print("Example 5: Reflection Before Execution (Error Demo)")
    print("=" * 70)

    chain = create_simple_chain()
    _ = ReasoningContext(  # Context not needed for this error demo
        outer_context=SAMPLE_FEEDBACK,
        api=create_openai_client(api_key="dummy", model="gpt-4"),
        model="unused",
        language=Language.ENGLISH,
    )

    print("\nAttempting to reflect before execution...")
    try:
        reflection = chain.reflect(task_description="This should fail")
        print(f"Unexpected success: {reflection}")
    except RuntimeError as e:
        print(f"Expected RuntimeError: {e}")
    print()


async def example_reflection_with_options():
    """Example 6: Reflection with custom options for minimal/focused analysis."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nSkipping: OPENAI_API_KEY not set")
        return

    client = create_openai_client(
        api_key=api_key,
        model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"),
        temperature=0.5,
    )

    context = ReasoningContext(
        outer_context=SAMPLE_FEEDBACK,
        api=client,
        model="unused",
        language=Language.ENGLISH,
        system_prompt="You are a customer experience analyst.",
    )

    chain = create_sentiment_chain()

    print("\n" + "=" * 70)
    print("Example 6: Reflection with Custom Options")
    print("=" * 70)

    # Execute the chain
    result = await chain.execute_async(context)

    print(f"\nExecution Status: {format_status(result.success)}")
    print(f"Total Steps: {len(result.step_results)}")

    # Example 6a: Minimal reflection (faster, cheaper)
    print("\n" + "-" * 70)
    print("6a. Minimal Reflection (excludes chain structure, dependency analysis)")
    print("-" * 70)

    minimal_options = ReflectionOptions(
        include_chain_structure=False,
        include_dependency_analysis=False,
        max_output_preview_chars=200,
        max_result_preview_chars=150,
    )

    reflection_minimal = chain.reflect(
        task_description="Analyze customer feedback",
        options=minimal_options,
    )
    print("\nMinimal Reflection:")
    print(reflection_minimal)

    # Example 6b: Full detailed reflection
    print("\n" + "-" * 70)
    print("6b. Full Detailed Reflection (all sections included)")
    print("-" * 70)

    full_options = ReflectionOptions(
        include_chain_structure=True,
        include_step_definitions=True,
        include_execution_metrics=True,
        include_dependency_analysis=True,
        max_output_preview_chars=1000,
    )

    reflection_full = chain.reflect(
        task_description="Analyze customer feedback and provide detailed recommendations",
        options=full_options,
    )
    print("\nFull Reflection:")
    print(reflection_full)
    print()


async def example_callbacks_and_cancellation():
    """Example 7: Demonstrating callbacks and cancellation support."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nSkipping: OPENAI_API_KEY not set")
        return

    client = create_openai_client(
        api_key=api_key,
        model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"),
        temperature=0.5,
    )

    print("\n" + "=" * 70)
    print("Example 7: Callbacks and Cancellation")
    print("=" * 70)

    # Track execution with callbacks
    step_events = []

    def on_step_start(step_number: int, step_title: str):
        step_events.append(f"START: Step {step_number} - {step_title}")
        print(f"  ▶ Starting step {step_number}: {step_title}")

    def on_step_complete(result):
        status = "✓" if result.success else "✗"
        step_events.append(f"COMPLETE: Step {result.step_number} - {status}")
        print(f"  ◀ Completed step {result.step_number}: {status}")

    def on_chain_complete(result):
        print(f"\n  Chain completed! Success: {result.success}")

    context = ReasoningContext(
        outer_context=SAMPLE_FEEDBACK,
        api=client,
        model="unused",
        language=Language.ENGLISH,
        on_step_start=on_step_start,
        on_step_complete=on_step_complete,
        on_chain_complete=on_chain_complete,
    )

    chain = create_simple_chain()

    print("\nExecuting with callbacks...")
    result = await chain.execute_async(context)

    print(f"\nTotal events captured: {len(step_events)}")
    print(f"Final status: {format_status(result.success)}")
    print()


async def main():
    """Run all reflection examples."""
    print("=" * 70)
    print("CARL Reflection Feature Examples")
    print("=" * 70)
    print("\nThese examples demonstrate the reflection functionality for analyzing")
    print("chain execution results after completion.")
    print("\nSet OPENAI_API_KEY to run these examples.")
    print("  export OPENAI_API_KEY='sk-or-v1-...'")

    # Run examples
    await example_basic_reflection()
    await example_async_reflection()
    await example_reflection_with_failures()
    await example_russian_reflection()
    await example_reflection_before_execution()
    await example_reflection_with_options()
    await example_callbacks_and_cancellation()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
