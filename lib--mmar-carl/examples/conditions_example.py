#!/usr/bin/env python3
"""
Example: Conditional Steps in CARL.

This example demonstrates conditional branching in reasoning chains:
- Built-in condition patterns (contains, equals, startswith, etc.)
- Complex expressions via simpleeval (len, comparisons, etc.)
- Routing execution flow based on tool results
- Using ChainBuilder for conditional steps
- Using typed ConditionalStepDescription directly

No LLM or API key required — all examples use tool and conditional steps only.

Usage:
    python examples/conditions_example.py
"""

import asyncio
import json

from mmar_carl import (
    ChainBuilder,
    ConditionalBranch,
    ConditionalStepConfig,
    ConditionalStepDescription,
    Language,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
    StepType,
    ToolStepConfig,
    ToolStepDescription,
    create_step,
)
from examples.utils import format_status


def find_step_result(result, step_number):
    """Find a step result by step number."""
    for sr in result.step_results:
        if sr.step_number == step_number:
            return sr
    return None


# Mock client — no LLM needed for these examples
class MockClient(LLMClientBase):
    """Minimal mock client for non-LLM chains."""

    async def get_response(self, prompt: str) -> str:
        return "mock"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


# ============================================================================
# Tool functions used in examples
# ============================================================================


def classify_text(text: str) -> str:
    """Classify text as positive, negative, or neutral based on keywords."""
    text_lower = text.lower()
    positive_words = {"great", "good", "excellent", "amazing", "love", "wonderful", "fantastic"}
    negative_words = {"bad", "terrible", "awful", "hate", "poor", "horrible", "worst"}

    pos_count = sum(1 for w in text_lower.split() if w in positive_words)
    neg_count = sum(1 for w in text_lower.split() if w in negative_words)

    if pos_count > neg_count:
        return "sentiment:positive"
    elif neg_count > pos_count:
        return "sentiment:negative"
    return "sentiment:neutral"


def handle_positive() -> str:
    """Handle positive sentiment."""
    return "Response: Thank you for the positive feedback! We're glad you're satisfied."


def handle_negative() -> str:
    """Handle negative sentiment."""
    return "Response: We're sorry to hear about your experience. Escalating to support team."


def handle_neutral() -> str:
    """Handle neutral sentiment."""
    return "Response: Thank you for your feedback. Is there anything specific we can help with?"


def get_score() -> int:
    """Return a sample score for threshold-based routing."""
    return 75


def format_score_high(score: int) -> str:
    """Format response for high score."""
    return f"Score {score}: Excellent! Above threshold."


def format_score_low(score: int) -> str:
    """Format response for low score."""
    return f"Score {score}: Below threshold. Needs improvement."


def get_user_role() -> str:
    """Return a sample user role."""
    return "admin"


def admin_action() -> str:
    """Perform admin action."""
    return "Admin panel: Full access granted. Showing all controls."


def editor_action() -> str:
    """Perform editor action."""
    return "Editor view: Content management tools loaded."


def viewer_action() -> str:
    """Perform viewer action."""
    return "Viewer mode: Read-only access. Displaying content."


# ============================================================================
# Example 1: Sentiment-based routing with built-in patterns
# ============================================================================


async def example_sentiment_routing():
    """
    Example 1: Route based on sentiment classification.

    Uses built-in 'contains:' pattern to branch on classifier output.
    Flow: classify text -> conditional branch -> handle sentiment
    """
    print("\n" + "=" * 60)
    print("Example 1: Sentiment-Based Routing (built-in patterns)")
    print("=" * 60)

    chain = (
        ChainBuilder()
        .add_tool_step(
            number=1,
            title="Classify Sentiment",
            tool_name="classify_text",
            input_mapping={"text": "$outer_context"},
        )
        .add_conditional_step(
            number=2,
            title="Route by Sentiment",
            condition_context_key="$metadata.step_1",
            branches=[
                ("contains:positive", 3),
                ("contains:negative", 4),
                ("contains:neutral", 5),
            ],
            default_step=5,
            dependencies=[1],
        )
        # Branch targets
        .add_tool_step(number=3, title="Handle Positive", tool_name="handle_positive", input_mapping={})
        .add_tool_step(number=4, title="Handle Negative", tool_name="handle_negative", input_mapping={})
        .add_tool_step(number=5, title="Handle Neutral", tool_name="handle_neutral", input_mapping={})
        .with_trace_name("Sentiment-Based Routing")
        .build()
    )

    print(f"Chain has {len(chain.steps)} steps")
    print(f"Execution plan: {chain.get_execution_plan()}")

    # Test with different inputs
    test_inputs = [
        "This product is great and amazing!",
        "Terrible experience, awful service.",
        "The package arrived on Tuesday.",
    ]

    for text in test_inputs:
        context = ReasoningContext(
            outer_context=text,
            api=MockClient(),
            model="unused",
            language=Language.ENGLISH,
        )
        context.register_tool("classify_text", classify_text)
        context.register_tool("handle_positive", handle_positive)
        context.register_tool("handle_negative", handle_negative)
        context.register_tool("handle_neutral", handle_neutral)

        result = await chain.execute_async(context)

        classify_result = find_step_result(result, 1)
        cond_result = find_step_result(result, 2)

        print(f'\n  Input: "{text}"')
        print(f"  Classification: {classify_result.result}")
        if isinstance(cond_result.result_data, dict):
            print(f"  Condition matched: {cond_result.result_data.get('matched_condition', 'N/A')}")
            print(f"  Routed to step: {cond_result.result_data.get('next_step', 'N/A')}")
        else:
            print(f"  Result: {cond_result.result}")
        print(f"  Execution: {format_status(result.success)}")


# ============================================================================
# Example 2: Threshold-based routing with complex expressions
# ============================================================================


async def example_threshold_routing():
    """
    Example 2: Route based on numeric threshold.

    Uses simpleeval expressions to evaluate numeric conditions.
    Flow: get score -> conditional (score >= 70?) -> format result
    """
    print("\n" + "=" * 60)
    print("Example 2: Threshold Routing (simpleeval expressions)")
    print("=" * 60)

    # Using typed step descriptions directly
    steps = [
        ToolStepDescription(
            number=1,
            title="Get Score",
            config=ToolStepConfig(tool_name="get_score", input_mapping={}),
        ),
        ConditionalStepDescription(
            number=2,
            title="Check Threshold",
            config=ConditionalStepConfig(
                condition_context_key="$metadata.step_1",
                branches=[
                    ConditionalBranch(condition="int(value) >= 70", next_step=3),
                    ConditionalBranch(condition="int(value) < 70", next_step=4),
                ],
                default_step=4,
            ),
            dependencies=[1],
        ),
        ToolStepDescription(
            number=3,
            title="High Score",
            config=ToolStepConfig(
                tool_name="format_score_high",
                input_mapping={"score": "$metadata.step_1"},
            ),
        ),
        ToolStepDescription(
            number=4,
            title="Low Score",
            config=ToolStepConfig(
                tool_name="format_score_low",
                input_mapping={"score": "$metadata.step_1"},
            ),
        ),
    ]

    chain = ReasoningChain(steps=steps, max_workers=2, trace_name="Threshold Routing")

    print(f"Chain has {len(chain.steps)} steps")
    print("Condition uses simpleeval: int(value) >= 70")

    context = ReasoningContext(
        outer_context="",
        api=MockClient(),
        model="unused",
        language=Language.ENGLISH,
    )
    context.register_tool("get_score", get_score)
    context.register_tool("format_score_high", format_score_high)
    context.register_tool("format_score_low", format_score_low)

    result = await chain.execute_async(context)

    score_result = find_step_result(result, 1)
    cond_result = find_step_result(result, 2)

    print(f"\n  Score: {score_result.result}")
    if isinstance(cond_result.result_data, dict):
        print(f"  Condition matched: {cond_result.result_data.get('matched_condition', 'N/A')}")
        print(f"  Routed to step: {cond_result.result_data.get('next_step', 'N/A')}")
    else:
        print(f"  Result: {cond_result.result}")
    print(f"  Execution: {format_status(result.success)}")


# ============================================================================
# Example 3: Role-based routing with equals pattern
# ============================================================================


async def example_role_routing():
    """
    Example 3: Route based on user role using equals pattern.

    Uses built-in 'equals:' pattern for exact string matching.
    Flow: get role -> conditional branch -> role-specific action
    """
    print("\n" + "=" * 60)
    print("Example 3: Role-Based Routing (equals pattern)")
    print("=" * 60)

    chain = (
        ChainBuilder()
        .add_tool_step(
            number=1,
            title="Get User Role",
            tool_name="get_user_role",
            input_mapping={},
        )
        .add_conditional_step(
            number=2,
            title="Route by Role",
            condition_context_key="$metadata.step_1",
            branches=[
                ("equals:admin", 3),
                ("equals:editor", 4),
                ("equals:viewer", 5),
            ],
            default_step=5,
            dependencies=[1],
        )
        .add_tool_step(number=3, title="Admin Action", tool_name="admin_action", input_mapping={})
        .add_tool_step(number=4, title="Editor Action", tool_name="editor_action", input_mapping={})
        .add_tool_step(number=5, title="Viewer Action", tool_name="viewer_action", input_mapping={})
        .with_trace_name("Role-Based Routing")
        .build()
    )

    print(f"Chain has {len(chain.steps)} steps")

    context = ReasoningContext(
        outer_context="",
        api=MockClient(),
        model="unused",
        language=Language.ENGLISH,
    )
    context.register_tool("get_user_role", get_user_role)
    context.register_tool("admin_action", admin_action)
    context.register_tool("editor_action", editor_action)
    context.register_tool("viewer_action", viewer_action)

    result = await chain.execute_async(context)

    role_result = find_step_result(result, 1)
    cond_result = find_step_result(result, 2)

    print(f"\n  Role: {role_result.result}")
    if isinstance(cond_result.result_data, dict):
        print(f"  Condition matched: {cond_result.result_data.get('matched_condition', 'N/A')}")
        print(f"  Routed to step: {cond_result.result_data.get('next_step', 'N/A')}")
    else:
        print(f"  Result: {cond_result.result}")
    print(f"  Execution: {format_status(result.success)}")


# ============================================================================
# Example 4: Using create_step() factory for conditional steps
# ============================================================================


def example_create_step_factory():
    """
    Example 4: Create conditional steps using the factory function.

    Demonstrates the create_step() API for building conditional steps.
    """
    print("\n" + "=" * 60)
    print("Example 4: create_step() Factory for Conditional Steps")
    print("=" * 60)

    cond_step = create_step(
        number=1,
        title="Check Value",
        step_type=StepType.CONDITIONAL,
        config=ConditionalStepConfig(
            condition_context_key="$history[-1]",
            branches=[
                ConditionalBranch(condition="contains:error", next_step=2),
                ConditionalBranch(condition="contains:success", next_step=3),
                ConditionalBranch(condition="empty", next_step=4),
            ],
            default_step=3,
        ),
    )

    print(f"Created step: {cond_step.title}")
    print(f"  Type: {cond_step.step_type}")
    print(f"  Branches: {len(cond_step.config.branches)}")
    for i, branch in enumerate(cond_step.config.branches, 1):
        print(f'    {i}. condition="{branch.condition}" -> step {branch.next_step}')
    print(f"  Default step: {cond_step.config.default_step}")

    # Demonstrate serialization
    step_dict = cond_step.model_dump()
    print(f"\n  Serialized to dict: {json.dumps(step_dict, indent=2)}")


# ============================================================================
# Example 5: All built-in condition patterns
# ============================================================================


async def example_all_patterns():
    """
    Example 5: Demonstrate all built-in condition patterns.

    Shows every supported condition pattern:
    - contains, equals, startswith, endswith, matches (regex)
    - empty, nonempty
    - Complex expressions via simpleeval
    """
    print("\n" + "=" * 60)
    print("Example 5: All Condition Patterns")
    print("=" * 60)

    # We'll test conditions directly by creating minimal chains
    test_cases = [
        ("contains:error", "An error occurred", True),
        ("contains:error", "All systems normal", False),
        ("equals:admin", "admin", True),
        ("equals:admin", "Admin", False),  # case-sensitive
        ("startswith:HTTP/", "HTTP/1.1 200 OK", True),
        ("endswith:.json", "config.json", True),
        ("matches:\\d{3}", "Status code 404 returned", True),
        ("matches:^\\d+$", "abc123", False),
        ("empty", "   ", True),
        ("empty", "hello", False),
        ("nonempty", "hello", True),
        ("nonempty", "  ", False),
        ("len(value) > 5", "hello world", True),
        ("len(value) > 5", "hi", False),
        ("int(value) >= 100", "150", True),
        ("'error' in value or 'fail' in value", "test fail case", True),
    ]

    print(f"\n{'Pattern':<45} {'Value':<25} {'Expected':<10} {'Result':<10}")
    print("-" * 90)

    for pattern, value, expected in test_cases:
        # Build a minimal chain to test the condition
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Value",
                config=ToolStepConfig(tool_name="provide_value", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Condition",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition=pattern, next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Matched",
                config=ToolStepConfig(tool_name="return_matched", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Not Matched",
                config=ToolStepConfig(tool_name="return_not_matched", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1, trace_name=f"Pattern Test: {pattern}")
        context = ReasoningContext(
            outer_context="",
            api=MockClient(),
            model="unused",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_value", lambda v=value: v)
        context.register_tool("return_matched", lambda: "matched")
        context.register_tool("return_not_matched", lambda: "not_matched")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)
        if isinstance(cond_result.result_data, dict):
            matched = cond_result.result_data.get("matched_condition", "(default)") != "(default)"
        else:
            matched = "(default)" not in cond_result.result
        status = "ok" if matched == expected else "MISMATCH"

        print(f"  {pattern:<43} {value:<25} {str(expected):<10} {status}")


# ============================================================================
# Main
# ============================================================================


async def main():
    """Run all conditional step examples."""
    print("CARL Conditional Steps Examples")
    print("=" * 60)
    print("\nConditional steps route execution flow based on conditions.")
    print("No LLM or API key required for these examples.\n")

    await example_sentiment_routing()
    await example_threshold_routing()
    await example_role_routing()
    example_create_step_factory()
    await example_all_patterns()

    print("\n" + "=" * 60)
    print("Conditional step examples completed!")
    print("=" * 60)
    print("\nBuilt-in condition patterns:")
    print("  contains:X      - check if value contains X")
    print("  equals:X        - check if value equals X (exact)")
    print("  startswith:X    - check if value starts with X")
    print("  endswith:X      - check if value ends with X")
    print("  matches:REGEX   - check if value matches regex")
    print("  empty           - check if value is empty/whitespace")
    print("  nonempty        - check if value is not empty")
    print("\nComplex expressions (via simpleeval):")
    print("  len(value) > 5            - length comparison")
    print("  int(value) >= 100         - numeric comparison")
    print("  'x' in value or 'y' in v  - logical operators")
    print("\nNext steps:")
    print("  - See tool_steps_example.py for tool/memory/transform steps")
    print("  - See basic_chain_example.py for chain fundamentals")
    print("  - See openrouter_example.py for LLM-powered chains")


if __name__ == "__main__":
    asyncio.run(main())
