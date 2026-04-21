#!/usr/bin/env python3
"""
Example: Metric-enhanced reflection in CARL.

This example runs WITHOUT an API key — all LLM responses are mocked.

Shows how metric scores and optional user-provided feedback are automatically
fed into the reflection prompt, giving the LLM concrete quality signals to
reference when suggesting chain improvements.

Three examples:
  1. Step metrics → reflection sees per-step scores
  2. Chain metrics + extra_feedback dict → combined context in prompt
  3. Russian language + extra_feedback string

Usage:
    python examples/reflection_metrics_example.py
"""

import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mmar_carl import (  # noqa: E402
    Language,
    LLMClientBase,
    LLMStepDescription,
    MetricBase,
    ReasoningChain,
    ReasoningContext,
    ReflectionOptions,
)


# ============================================================
# Mock LLM client
# ============================================================


class MockClient(LLMClientBase):
    """
    Returns deterministic step outputs and a canned reflection.

    The reflection response deliberately references metric values so the
    printed output clearly demonstrates that the LLM received the scores.
    """

    STEP_RESPONSES: dict[str, str] = {
        "data overview": (
            "The dataset covers Q1–Q4 2023. 12 records, no missing values. Revenue ranges from $1.2M to $3.8M."
        ),
        "trend analysis": (
            "Revenue grew 217% year-over-year in 2023. "
            "Q1: $1.2M, Q2: $1.9M, Q3: $2.7M, Q4: $3.8M. "
            "Compound monthly growth rate: 10.3%. "
            "Seasonal peaks align with product launches."
        ),
        "risk": ("Three risks: supplier concentration, capacity constraints, competition. Overall risk: medium."),
        "executive summary": (
            "Strong 2023 performance. Recommend supplier diversification and "
            "operational investment. Monitor competition."
        ),
    }

    REFLECTION_TEMPLATE = (
        "Based on the execution results and metric scores provided:\n"
        "- Step 1 (Data Overview) received word_count={wc:.0f} — adequate length.\n"
        "- Step 2 (Trend Analysis) keyword_coverage={kc:.2f} — "
        "{kc_comment}.\n"
        "- Chain-level llm_judge_score={judge:.1f}/10 — "
        "the final output meets quality expectations.\n\n"
        "Recommended improvement: add a 'Competitor Benchmarking' step between "
        "Risk Assessment and Executive Summary to improve coverage."
    )

    def __init__(self):
        self._reflection_mode = False
        self._last_kc: float = 0.0
        self._last_wc: float = 0.0
        self._last_judge: float = 0.0

    def set_reflection_mode(self, word_count: float, keyword_cov: float, judge: float) -> None:
        self._reflection_mode = True
        self._last_wc = word_count
        self._last_kc = keyword_cov
        self._last_judge = judge

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        if self._reflection_mode:
            kc_comment = "good coverage" if self._last_kc >= 0.8 else "could add more revenue terms"
            return self.REFLECTION_TEMPLATE.format(
                wc=self._last_wc,
                kc=self._last_kc,
                kc_comment=kc_comment,
                judge=self._last_judge,
            )
        lowered = prompt.lower()
        for key, resp in self.STEP_RESPONSES.items():
            if key in lowered:
                return resp
        return "Analysis complete."


# ============================================================
# Metric implementations
# ============================================================


class WordCountMetric(MetricBase):
    @property
    def name(self) -> str:
        return "word_count"

    async def compute_async(self, text: str) -> float:
        return float(len(text.split()))


class KeywordCoverageMetric(MetricBase):
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


class MockJudgeMetric(MetricBase):
    """Simulates LLM-as-a-judge: score 0–10 based on word count."""

    def __init__(self, min_words: int = 10, max_words: int = 80):
        self._min = min_words
        self._max = max_words

    @property
    def name(self) -> str:
        return "llm_judge_score"

    async def compute_async(self, text: str) -> float:
        words = len(text.split())
        clamped = max(self._min, min(self._max, words))
        return round((clamped - self._min) / (self._max - self._min) * 10, 2)


# ============================================================
# Helpers
# ============================================================


def separator(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def show_metrics(label: str, scores: dict) -> None:
    if not scores:
        print(f"  {label}: (none)")
        return
    print(f"  {label}:")
    for name, value in scores.items():
        print(f"    {name}: {value}")


def show_prompt_excerpt(prompt: str, keywords: list[str]) -> None:
    """Print lines from the prompt that contain any of the keywords."""
    relevant = [line for line in prompt.splitlines() if any(kw in line for kw in keywords)]
    if relevant:
        print("  Relevant lines in reflection prompt:")
        for line in relevant:
            print(f"    > {line.strip()}")


# ============================================================
# Example 1: Step metrics feed into reflection
# ============================================================


async def example_step_metrics_in_reflection() -> None:
    separator("Example 1: Step metric scores in reflection prompt")

    client = MockClient()

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
                KeywordCoverageMetric(["growth", "revenue", "quarterly", "seasonal"]),
                WordCountMetric(),
            ],
        ),
        LLMStepDescription(
            number=3,
            title="Risk Assessment",
            aim="Identify key business risks",
            dependencies=[1],
            metrics=[KeywordCoverageMetric(["risk", "supply", "competition"])],
        ),
    ]

    chain = ReasoningChain(steps=steps, max_workers=2)
    context = ReasoningContext(
        outer_context="Quarterly revenue 2023",
        api=client,
        model="mock",
        language=Language.ENGLISH,
    )

    result = await chain.execute_async(context)
    print(f"\nExecution: {'OK' if result.success else 'FAILED'}")

    print("\nPer-step metric scores:")
    for sr in result.step_results:
        show_metrics(f"Step {sr.step_number} ({sr.step_title})", sr.metrics)

    # Switch to reflection mode
    wc = result.step_results[0].metrics.get("word_count", 20)
    kc = result.step_results[1].metrics.get("keyword_coverage", 0.5)
    client.set_reflection_mode(word_count=wc, keyword_cov=kc, judge=8.0)

    # Reflect — metric scores are included by default (include_metric_scores=True)
    reflection = await chain.reflect_async(task_description="Analyse quarterly revenue data")
    print("\nReflection (excerpt):")
    print(reflection[:500])

    # Show that the prompt contained the scores
    print(
        "\n[Note: ReflectionOptions.include_metric_scores=True by default — "
        "metric scores were sent to the LLM in the prompt]"
    )


# ============================================================
# Example 2: Chain metrics + extra_feedback dict
# ============================================================


async def example_chain_metrics_and_extra_feedback() -> None:
    separator("Example 2: Chain metrics + extra_feedback dict")

    client = MockClient()

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

    chain = ReasoningChain(
        steps=steps,
        metrics=[
            MockJudgeMetric(min_words=10, max_words=80),
            KeywordCoverageMetric(["growth", "revenue", "recommend"]),
        ],
    )

    context = ReasoningContext(
        outer_context="Annual revenue data",
        api=client,
        model="mock",
        language=Language.ENGLISH,
    )

    result = await chain.execute_async(context)
    print(f"\nExecution: {'OK' if result.success else 'FAILED'}")
    show_metrics("Chain metric scores", result.metrics)

    judge = result.metrics.get("llm_judge_score", 7.0)
    kc = result.metrics.get("keyword_coverage", 0.7)
    client.set_reflection_mode(word_count=35, keyword_cov=kc, judge=judge)

    # Pass extra context that the LLM should consider
    options = ReflectionOptions(
        include_metric_scores=True,
        extra_feedback={
            "domain": "financial analysis",
            "audience": "C-level executives",
            "priority": "conciseness over detail",
        },
    )

    reflection = await chain.reflect_async(
        task_description="Write an executive summary of revenue trends",
        options=options,
    )
    print("\nReflection:")
    print(reflection[:500])
    print(
        "\n[Note: extra_feedback dict was appended to the reflection prompt as "
        "labelled entries — the LLM sees domain, audience, and priority]"
    )


# ============================================================
# Example 3: Russian language + extra_feedback string
# ============================================================


async def example_russian_with_feedback_string() -> None:
    separator("Example 3: Russian language + extra_feedback string")

    client = MockClient()

    steps = [
        LLMStepDescription(
            number=1,
            title="Risk Assessment",
            aim="Оценить основные риски бизнеса",
            metrics=[KeywordCoverageMetric(["риск", "поставщик", "конкуренция"])],
        ),
    ]

    chain = ReasoningChain(steps=steps, max_workers=1)
    context = ReasoningContext(
        outer_context="Данные о доходах за 2023 год",
        api=client,
        model="mock",
        language=Language.RUSSIAN,
    )

    result = await chain.execute_async(context)
    print(f"\nВыполнение: {'ОК' if result.success else 'ОШИБКА'}")
    show_metrics("Метрики шага 1", result.step_results[0].metrics if result.step_results else {})

    client.set_reflection_mode(word_count=20, keyword_cov=0.33, judge=5.0)

    # String extra_feedback — freeform notes
    options = ReflectionOptions(
        include_metric_scores=True,
        extra_feedback=("Приоритет: улучшить охват ключевых слов по рискам. Аудитория: совет директоров."),
    )

    reflection = await chain.reflect_async(
        task_description="Оценить риски для финансового отчёта",
        options=options,
    )
    print("\nРефлексия:")
    print(reflection[:500])
    print("\n[Примечание: extra_feedback — строка, заголовок раздела на русском: 'Дополнительный контекст']")


# ============================================================
# Main
# ============================================================


async def main() -> None:
    print("CARL — Metric-Enhanced Reflection Examples")
    print("No API key required — using mock LLM client\n")
    print("Demonstrates how MetricBase scores and extra_feedback are")
    print("fed into the reflection prompt for richer LLM analysis.")

    await example_step_metrics_in_reflection()
    await example_chain_metrics_and_extra_feedback()
    await example_russian_with_feedback_string()

    print("\n" + "=" * 65)
    print("All reflection-metrics examples completed!")
    print("=" * 65)
    print("\nKey takeaways:")
    print("  - ReflectionOptions.include_metric_scores=True (default) feeds all")
    print("    MetricBase scores into the reflection prompt automatically")
    print("  - ReflectionOptions.extra_feedback accepts a dict or string of")
    print("    user-provided context appended to the reflection prompt")
    print("  - Both are optional — set include_metric_scores=False or leave")
    print("    extra_feedback=None to omit them")


if __name__ == "__main__":
    asyncio.run(main())
