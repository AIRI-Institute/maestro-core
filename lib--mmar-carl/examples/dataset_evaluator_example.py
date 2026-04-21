#!/usr/bin/env python3
"""
Example: Dataset evaluation with SimpleDataset and DatasetEvaluator.

This example runs WITHOUT an API key — all LLM responses are mocked.

Demonstrates:
  1. Building a SimpleDataset from a list of DataCase objects
  2. Running DatasetEvaluator to score every case with a metric
  3. Using TopKWorstStrategy to surface problem cases
  4. Using ThresholdStrategy as an alternative
  5. Passing the report to chain reflection via dataset_report
  6. batch_execute — run a chain on multiple contexts at once

Usage:
    python examples/dataset_evaluator_example.py
"""

import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mmar_carl import (  # noqa: E402
    DataCase,
    DatasetEvaluator,
    LLMClientBase,
    LLMStepDescription,
    MetricBase,
    ReasoningChain,
    ReasoningContext,
    SimpleDataset,
    TopKWorstStrategy,
    ThresholdStrategy,
)


# ============================================================
# Mock LLM client — deterministic, no API key needed
# ============================================================

# Maps keywords in outer_context → mock output.
# The "bad" cases intentionally produce short, low-quality outputs
# so the metric can distinguish them.
_MOCK_RESPONSES: dict[str, str] = {
    "quarterly revenue": (
        "Revenue grew from $1.2M in Q1 to $3.8M in Q4, a 217% increase. "
        "Key drivers were the new product line and market expansion. "
        "Risks include supply chain concentration and competitive pressure."
    ),
    "annual headcount": (
        "Headcount increased from 45 to 112 employees, driven by engineering and sales hiring. "
        "Attrition remained below 8% throughout the year. "
        "Recommend: invest in onboarding to sustain retention as team scales."
    ),
    "customer churn": ("Churn rate: 2.1%. Short."),
    "marketing spend": ("Ok."),
    "product roadmap": (
        "Three major initiatives planned: platform migration, mobile app launch, and API expansion. "
        "The platform migration carries the highest technical risk but also the largest upside. "
        "Estimated delivery: Q3 for migration, Q4 for mobile, Q1 next year for API."
    ),
    "operational costs": (
        "Costs rose 34% year-over-year, in line with headcount growth. "
        "Cloud infrastructure is the fastest-growing line item at +67%. "
        "Optimisation opportunities exist in compute reservation and data egress."
    ),
}


class MockAnalystClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        lowered = prompt.lower()
        for key, response in _MOCK_RESPONSES.items():
            if key in lowered:
                return response
        return "Analysis complete."


# ============================================================
# Metric: keyword coverage (fraction of required words present)
# ============================================================


class KeywordCoverageMetric(MetricBase):
    """Score = fraction of required keywords present in the output (0-1)."""

    def __init__(self, keywords: list[str]) -> None:
        self._keywords = [k.lower() for k in keywords]

    @property
    def name(self) -> str:
        return "keyword_coverage"

    async def compute_async(self, output: Any) -> float:
        text = output.get_final_output().lower() if hasattr(output, "get_final_output") else str(output).lower()
        if not self._keywords:
            return 1.0
        hits = sum(1 for kw in self._keywords if kw in text)
        return hits / len(self._keywords)


# ============================================================
# Build chain and dataset
# ============================================================


def build_chain() -> ReasoningChain:
    steps = [
        LLMStepDescription(
            number=1,
            title="analysis",
            aim="Provide a concise analytical summary of the topic",
        )
    ]
    return ReasoningChain(steps=steps)


def build_dataset() -> SimpleDataset:
    cases = [
        DataCase(input="quarterly revenue figures for 2023", label="revenue"),
        DataCase(input="annual headcount changes", label="headcount"),
        DataCase(input="customer churn metrics", label="churn"),  # intentionally bad output
        DataCase(input="marketing spend breakdown", label="marketing"),  # intentionally bad output
        DataCase(input="product roadmap priorities", label="roadmap"),
        DataCase(input="operational costs analysis", label="opex"),
    ]
    return SimpleDataset(cases)


def make_context(case: DataCase) -> ReasoningContext:
    return ReasoningContext(
        outer_context=case.input,
        api=MockAnalystClient(),
    )


# ============================================================
# Section 1: DatasetEvaluator with TopKWorstStrategy
# ============================================================


def section_top_k_worst(chain: ReasoningChain, dataset: SimpleDataset) -> None:
    print("=" * 60)
    print("Section 1: TopKWorstStrategy (k=2)")
    print("=" * 60)

    metric = KeywordCoverageMetric(keywords=["risk", "recommend", "growth"])
    strategy = TopKWorstStrategy(k=2)

    evaluator = DatasetEvaluator(
        chain=chain,
        dataset=dataset,
        metric=metric,
        strategy=strategy,
    )

    report = evaluator.evaluate(context_factory=make_context)

    print(f"\nMetric      : {report.metric_name}")
    print(f"Cases total : {len(report.all_results)}")
    print(f"Mean score  : {report.mean_score:.3f}")
    print(f"Min / Max   : {report.min_score:.3f} / {report.max_score:.3f}")

    print("\nAll case scores:")
    for r in report.all_results:
        label = r.case.label or r.case.input[:25]
        flag = " ← problem" if r in report.selected_cases else ""
        print(f"  [{label:12s}]  score={r.score:.3f}  output={r.chain_output[:60]!r}{flag}")

    print(f"\nSelected problem cases ({len(report.selected_cases)}):")
    for r in report.selected_cases:
        print(f"  [{r.case.label}]  score={r.score:.3f}")


# ============================================================
# Section 2: ThresholdStrategy
# ============================================================


def section_threshold(chain: ReasoningChain, dataset: SimpleDataset) -> None:
    print("\n" + "=" * 60)
    print("Section 2: ThresholdStrategy (threshold=0.5)")
    print("=" * 60)

    metric = KeywordCoverageMetric(keywords=["risk", "recommend", "growth"])
    strategy = ThresholdStrategy(threshold=0.5, higher_is_better=True)

    evaluator = DatasetEvaluator(
        chain=chain,
        dataset=dataset,
        metric=metric,
        strategy=strategy,
    )

    report = evaluator.evaluate(context_factory=make_context)

    print(f"\nCases below threshold: {len(report.selected_cases)}")
    for r in report.selected_cases:
        print(f"  [{r.case.label}]  score={r.score:.3f}")


# ============================================================
# Section 3: Feed report into reflection
# ============================================================


def section_reflection(chain: ReasoningChain, dataset: SimpleDataset) -> None:
    print("\n" + "=" * 60)
    print("Section 3: Report → ReflectionOptions (dataset_report)")
    print("=" * 60)

    metric = KeywordCoverageMetric(keywords=["risk", "recommend", "growth"])
    strategy = TopKWorstStrategy(k=2)

    evaluator = DatasetEvaluator(
        chain=chain,
        dataset=dataset,
        metric=metric,
        strategy=strategy,
    )
    report = evaluator.evaluate(context_factory=make_context)

    # to_reflection_dict() is the lightweight path when you want to pass
    # the report as extra_feedback rather than using the dataset_report field.
    reflection_dict = report.to_reflection_dict(max_preview_chars=80)
    print("\nto_reflection_dict() output:")
    for k, v in reflection_dict.items():
        print(f"  {k}: {v!r}")


# ============================================================
# Section 4: batch_execute
# ============================================================


def section_batch_execute(chain: ReasoningChain, dataset: SimpleDataset) -> None:
    print("\n" + "=" * 60)
    print("Section 4: batch_execute on all dataset cases")
    print("=" * 60)

    contexts = [make_context(case) for case in dataset]
    results = chain.batch_execute(contexts)

    print(f"\nRan {len(results)} contexts:")
    for i, (case, result) in enumerate(zip(dataset, results)):
        label = case.label or f"case_{i + 1}"
        output_preview = result.get_final_output()[:60] if result.success else f"ERROR: {result.error}"
        print(f"  [{label:12s}]  success={result.success}  output={output_preview!r}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    chain = build_chain()
    dataset = build_dataset()

    section_top_k_worst(chain, dataset)
    section_threshold(chain, dataset)
    section_reflection(chain, dataset)
    section_batch_execute(chain, dataset)

    print("\nDone.")
