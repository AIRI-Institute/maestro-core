#!/usr/bin/env python3
"""
Example: Evaluation metrics for steps and chains.

This example runs WITHOUT an API key — all LLM responses are mocked.

Demonstrates four ready-to-use metric types:
  1. WordCountMetric     — counts words in the output
  2. SentenceLengthMetric — average sentence length
  3. KeywordCoverageMetric — fraction of required keywords found
  4. MockLLMJudgeMetric  — simulates an LLM-as-a-judge score (0–10)

Then shows how to attach them to individual steps and to the whole chain.

Usage:
    python examples/metrics_example.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mmar_carl import (  # noqa: E402
    LLMClientBase,
    LLMStepDescription,
    Language,
    MetricBase,
    ReasoningChain,
    ReasoningContext,
)


# ============================================================
# Mock LLM client — deterministic, no API key needed
# ============================================================


class MockAnalystClient(LLMClientBase):
    """
    Returns realistic-looking analysis text so metrics have meaningful input.
    Each step gets a slightly different response based on the step title.
    """

    RESPONSES: dict[str, str] = {
        "data overview": (
            "The dataset contains quarterly revenue figures for 2023. "
            "Total entries: 12. No missing values detected. "
            "Revenue ranges from $1.2M to $3.8M across quarters."
        ),
        "trend analysis": (
            "Revenue shows an upward trend throughout 2023. "
            "Q1: $1.2M. Q2: $1.9M. Q3: $2.7M. Q4: $3.8M. "
            "Year-over-year growth: 217%. The compound monthly growth rate is 10.3%. "
            "Seasonal peaks correlate with product launch cycles."
        ),
        "risk assessment": (
            "Three primary risks identified. "
            "First: dependency on a single supplier creates supply chain vulnerability. "
            "Second: rapid growth may strain operational capacity. "
            "Third: increasing competition in the core market segment. "
            "Overall risk level: medium."
        ),
        "executive summary": (
            "Strong financial performance in 2023 with 217% revenue growth. "
            "Key drivers: new product line and market expansion. "
            "Recommend: diversify supplier base, invest in operational scaling, "
            "and monitor competitive landscape closely."
        ),
    }

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        lowered = prompt.lower()
        for key, response in self.RESPONSES.items():
            if key in lowered:
                return response
        return "Analysis complete. No specific pattern matched."


# ============================================================
# Metric definitions
# ============================================================


class WordCountMetric(MetricBase):
    """
    Counts the number of words in the output.

    Useful as a basic length sanity check — very short outputs
    often indicate truncation or errors.
    """

    @property
    def name(self) -> str:
        return "word_count"

    async def compute_async(self, text: str) -> float:
        return float(len(text.split()))


class SentenceLengthMetric(MetricBase):
    """
    Computes the average sentence length (words per sentence).

    Higher values may indicate denser, more analytical writing.
    Lower values suggest bullet-point or list-style output.
    """

    @property
    def name(self) -> str:
        return "avg_sentence_length"

    async def compute_async(self, text: str) -> float:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not sentences:
            return 0.0
        total_words = sum(len(s.split()) for s in sentences)
        return round(total_words / len(sentences), 2)


class KeywordCoverageMetric(MetricBase):
    """
    Fraction of required keywords present in the output (0.0 – 1.0).

    Pass a list of domain-specific terms that must appear in a good answer.
    A score of 1.0 means all keywords were found; 0.0 means none.
    """

    def __init__(self, keywords: list[str]):
        self._keywords = [kw.lower() for kw in keywords]

    @property
    def name(self) -> str:
        return "keyword_coverage"

    async def compute_async(self, text: str) -> float:
        if not self._keywords:
            return 1.0
        lowered = text.lower()
        found = sum(1 for kw in self._keywords if kw in lowered)
        return round(found / len(self._keywords), 3)


class MockLLMJudgeMetric(MetricBase):
    """
    Simulates an LLM-as-a-judge metric (score 0–10).

    In production replace the `_judge` call with a real LLM call.
    Here the "judge" uses a simple heuristic: word count maps to 0–10.
    This demonstrates that a metric can be async and involve I/O.
    """

    def __init__(self, min_words: int = 10, max_words: int = 80):
        self._min = min_words
        self._max = max_words

    @property
    def name(self) -> str:
        return "llm_judge_score"

    async def compute_async(self, text: str) -> float:
        words = len(text.split())
        clamped = max(self._min, min(self._max, words))
        score = (clamped - self._min) / (self._max - self._min) * 10
        return round(score, 2)


# ============================================================
# Example 1: Metrics on individual steps
# ============================================================


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_step_metrics(step_number: int, step_title: str, metrics: dict) -> None:
    if not metrics:
        print(f"  Step {step_number} ({step_title}): no metrics")
        return
    print(f"  Step {step_number} ({step_title}):")
    for name, value in metrics.items():
        print(f"    {name}: {value}")


async def example_step_metrics() -> None:
    print_section("Example 1: Metrics on individual steps")

    steps = [
        LLMStepDescription(
            number=1,
            title="Data Overview",
            aim="Summarise the dataset",
            metrics=[
                WordCountMetric(),
                SentenceLengthMetric(),
            ],
        ),
        LLMStepDescription(
            number=2,
            title="Trend Analysis",
            aim="Identify revenue trends",
            dependencies=[1],
            metrics=[
                WordCountMetric(),
                KeywordCoverageMetric(["growth", "revenue", "quarterly"]),
                MockLLMJudgeMetric(min_words=15, max_words=60),
            ],
        ),
        LLMStepDescription(
            number=3,
            title="Risk Assessment",
            aim="Identify key risks",
            dependencies=[1],
            metrics=[
                KeywordCoverageMetric(["risk", "supply", "competition"]),
            ],
        ),
    ]

    chain = ReasoningChain(steps=steps, max_workers=2)
    context = ReasoningContext(
        outer_context="Quarterly revenue data 2023",
        api=MockAnalystClient(),
        model="mock",
        language=Language.ENGLISH,
    )

    result = await chain.execute_async(context)

    print(f"\nExecution: {'OK' if result.success else 'FAILED'}")
    print(f"Steps completed: {len(result.get_successful_steps())}/{len(steps)}")
    print("\nPer-step metric scores:")
    for sr in result.step_results:
        print_step_metrics(sr.step_number, sr.step_title, sr.metrics)


# ============================================================
# Example 2: Metrics on the chain (evaluated on final output)
# ============================================================


async def example_chain_metrics() -> None:
    print_section("Example 2: Metrics on the whole chain")

    steps = [
        LLMStepDescription(
            number=1,
            title="Trend Analysis",
            aim="Analyse quarterly trends",
        ),
        LLMStepDescription(
            number=2,
            title="Executive Summary",
            aim="Write an executive summary",
            dependencies=[1],
        ),
    ]

    # Chain-level metrics run on the final step's output
    chain = ReasoningChain(
        steps=steps,
        metrics=[
            WordCountMetric(),
            SentenceLengthMetric(),
            KeywordCoverageMetric(["growth", "revenue", "recommend"]),
            MockLLMJudgeMetric(min_words=20, max_words=100),
        ],
    )

    context = ReasoningContext(
        outer_context="Annual revenue data",
        api=MockAnalystClient(),
        model="mock",
        language=Language.ENGLISH,
    )

    result = await chain.execute_async(context)

    print(f"\nExecution: {'OK' if result.success else 'FAILED'}")
    print("\nChain-level metric scores (on final output):")
    for name, value in result.metrics.items():
        print(f"  {name}: {value}")


# ============================================================
# Example 3: Step + chain metrics combined
# ============================================================


async def example_combined_metrics() -> None:
    print_section("Example 3: Step metrics + chain metrics combined")

    steps = [
        LLMStepDescription(
            number=1,
            title="Data Overview",
            aim="Summarise the dataset",
            metrics=[WordCountMetric()],
        ),
        LLMStepDescription(
            number=2,
            title="Trend Analysis",
            aim="Identify revenue trends",
            dependencies=[1],
            metrics=[
                KeywordCoverageMetric(["growth", "revenue"]),
                MockLLMJudgeMetric(),
            ],
        ),
        LLMStepDescription(
            number=3,
            title="Executive Summary",
            aim="Write an executive summary",
            dependencies=[2],
            metrics=[WordCountMetric(), SentenceLengthMetric()],
        ),
    ]

    chain = ReasoningChain(
        steps=steps,
        metrics=[
            # Chain metric runs on the final step output
            MockLLMJudgeMetric(min_words=10, max_words=60),
            KeywordCoverageMetric(["recommend", "growth", "revenue"]),
        ],
    )

    context = ReasoningContext(
        outer_context="Quarterly revenue data",
        api=MockAnalystClient(),
        model="mock",
        language=Language.ENGLISH,
    )

    result = await chain.execute_async(context)

    print(f"\nExecution: {'OK' if result.success else 'FAILED'}")
    print("\nPer-step metrics:")
    for sr in result.step_results:
        print_step_metrics(sr.step_number, sr.step_title, sr.metrics)

    print("\nChain-level metrics (on final output):")
    for name, value in result.metrics.items():
        print(f"  {name}: {value}")

    print("\nResult dict sample (metrics field):")
    d = result.to_dict()
    print(f"  result['metrics'] = {d['metrics']}")
    print(f"  result['step_results'][0]['metrics'] = {d['step_results'][0]['metrics']}")


# ============================================================
# Main
# ============================================================


async def main() -> None:
    import asyncio  # noqa: F401 — used by the caller

    print("CARL Metrics Examples")
    print("No API key required — using mock LLM client")

    await example_step_metrics()
    await example_chain_metrics()
    await example_combined_metrics()

    print("\n" + "=" * 60)
    print("All metrics examples completed!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  - Attach metrics=[...] to LLMStepDescription for per-step scores")
    print("  - Attach metrics=[...] to ReasoningChain for chain-level scores")
    print("  - Scores are in StepExecutionResult.metrics and ReasoningResult.metrics")
    print("  - A failing metric never aborts execution")
    print("  - Implement MetricBase to add any custom evaluation logic")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
