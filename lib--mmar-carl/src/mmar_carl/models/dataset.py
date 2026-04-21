"""
Dataset abstractions for batch evaluation in CARL.

Provides DataCase, AbstractDataset, SimpleDataset, selection strategies,
CaseEvaluationResult, and DatasetEvaluationReport for use with DatasetEvaluator.
"""

from abc import ABC, abstractmethod
from typing import Annotated, Any, Iterator, Literal

from pydantic import BaseModel, Field, field_validator


class DataCase(BaseModel):
    """A single evaluation case for dataset-based chain assessment."""

    input: str = Field(..., description="Input data (outer_context) for chain execution")
    label: str | None = Field(
        default=None,
        description="Human-readable identifier for this case (e.g. 'case_01')",
    )
    expected: str | None = Field(
        default=None,
        description=(
            "Expected output for reference (optional, not used in metric computation)"
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional case-specific metadata",
    )


class AbstractDataset(ABC):
    """
    Abstract base class for evaluation datasets.

    Implement :meth:`__iter__` to yield :class:`DataCase` objects.
    Override :meth:`__len__` whenever the size is cheaply known.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[DataCase]: ...

    def __len__(self) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement __len__. "
            "Override it if the size is cheaply available."
        )


class SimpleDataset(AbstractDataset):
    """Concrete dataset backed by an in-memory list of :class:`DataCase` objects."""

    def __init__(self, cases: list[DataCase]) -> None:
        self._cases = list(cases)

    def __iter__(self) -> Iterator[DataCase]:
        return iter(self._cases)

    def __len__(self) -> int:
        return len(self._cases)


class DataFrameDataset(AbstractDataset):
    """
    Dataset backed by a ``pandas.DataFrame``.

    Requires pandas to be installed (``pip install pandas`` or
    ``pip install mmar-carl[pandas]``).

    Args:
        df: Source DataFrame.
        input_col: Column name whose values become :attr:`DataCase.input`.
        label_col: Optional column to use as :attr:`DataCase.label`.
        expected_col: Optional column to use as :attr:`DataCase.expected`.
        metadata_cols: Additional columns to include in :attr:`DataCase.metadata`.

    Example::

        import pandas as pd
        from mmar_carl import DataFrameDataset

        df = pd.DataFrame({
            "text": ["analyse Q1 revenue", "summarise risks"],
            "id": ["q1", "risks"],
        })
        dataset = DataFrameDataset(df, input_col="text", label_col="id")
    """

    def __init__(
        self,
        df: Any,
        *,
        input_col: str = "input",
        label_col: str | None = None,
        expected_col: str | None = None,
        metadata_cols: list[str] | None = None,
    ) -> None:
        try:
            import pandas  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pandas is required for DataFrameDataset. "
                "Install it with: pip install pandas  (or: pip install mmar-carl[pandas])"
            ) from exc

        if input_col not in df.columns:
            raise ValueError(f"Column '{input_col}' not found in DataFrame. Available: {list(df.columns)}")

        self._df = df
        self._input_col = input_col
        self._label_col = label_col
        self._expected_col = expected_col
        self._metadata_cols: list[str] = metadata_cols or []

    def __iter__(self) -> Iterator[DataCase]:
        for _, row in self._df.iterrows():
            meta = {col: row[col] for col in self._metadata_cols if col in self._df.columns}
            yield DataCase(
                input=str(row[self._input_col]),
                label=str(row[self._label_col]) if self._label_col is not None else None,
                expected=str(row[self._expected_col]) if self._expected_col is not None else None,
                metadata=meta,
            )

    def __len__(self) -> int:
        return len(self._df)


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------


class ThresholdStrategy(BaseModel):
    """
    Select cases where the metric score crosses a fixed threshold.

    With ``higher_is_better=True`` (default), selects cases where
    ``score < threshold``.  With ``higher_is_better=False``, selects cases
    where ``score > threshold``.
    """

    mode: Literal["threshold"] = "threshold"
    threshold: float = Field(..., description="Score boundary for case selection")
    higher_is_better: bool = Field(
        default=True,
        description=(
            "Metric direction.  True → lower scores are worse (select below threshold)."
            "  False → higher scores are worse (select above threshold)."
        ),
    )

    def select(
        self, results: "list[CaseEvaluationResult]"
    ) -> "list[CaseEvaluationResult]":
        """Return cases that fail the threshold criterion."""
        if not results:
            return []
        if self.higher_is_better:
            return [r for r in results if r.score < self.threshold]
        else:
            return [r for r in results if r.score > self.threshold]


class TopKWorstStrategy(BaseModel):
    """
    Select the *k* worst-scoring cases.

    When ``include_ties=True`` (default), all cases tied at the k-th boundary
    score are included, so the actual count may exceed *k*.
    """

    mode: Literal["top_k_worst"] = "top_k_worst"
    k: int = Field(..., gt=0, description="Number of worst cases to select (must be > 0)")
    higher_is_better: bool = Field(
        default=True,
        description="Metric direction.  Determines what 'worst' means.",
    )
    include_ties: bool = Field(
        default=True,
        description=(
            "Include all cases tied at the k-th boundary score.  "
            "Actual result count may be > k."
        ),
    )

    @field_validator("k")
    @classmethod
    def _k_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("k must be greater than 0")
        return v

    def select(
        self, results: "list[CaseEvaluationResult]"
    ) -> "list[CaseEvaluationResult]":
        """Return the k worst cases (ascending score for higher_is_better metrics)."""
        if not results:
            return []

        # Sort worst-first
        sorted_results = sorted(
            results,
            key=lambda r: r.score,
            reverse=not self.higher_is_better,
        )

        if self.k >= len(sorted_results):
            return list(sorted_results)

        if self.include_ties:
            boundary_score = sorted_results[self.k - 1].score
            if self.higher_is_better:
                return [r for r in sorted_results if r.score <= boundary_score]
            else:
                return [r for r in sorted_results if r.score >= boundary_score]

        return sorted_results[: self.k]


# Discriminated union — Pydantic uses the `mode` literal to pick the right model.
SelectionStrategy = Annotated[
    ThresholdStrategy | TopKWorstStrategy,
    Field(discriminator="mode"),
]


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------


class CaseEvaluationResult(BaseModel):
    """Result of running a :class:`~mmar_carl.chain.ReasoningChain` on a single :class:`DataCase`."""

    case: DataCase
    score: float = Field(..., description="Metric score for this case")
    chain_output: str = Field(..., description="Final output from chain execution")
    success: bool = Field(..., description="Whether the chain executed successfully")
    execution_time: float | None = Field(
        default=None, description="Chain execution time in seconds"
    )


class DatasetEvaluationReport(BaseModel):
    """
    Aggregated result of evaluating a dataset against a chain.

    Pass this to :class:`~mmar_carl.chain.ReflectionOptions` via the
    ``dataset_report`` field to include a dedicated problem-cases section in the
    reflection prompt:

    .. code-block:: python

        options = ReflectionOptions(dataset_report=report)
        reflection = chain.reflect("task description", options=options)

    Alternatively, use :meth:`to_reflection_dict` for the MVP path via
    ``extra_feedback``.
    """

    metric_name: str = Field(..., description="Name of the metric used for evaluation")
    strategy: SelectionStrategy = Field(..., description="Selection strategy applied")
    all_results: list[CaseEvaluationResult] = Field(
        ..., description="Evaluation results for every case in the dataset"
    )
    selected_cases: list[CaseEvaluationResult] = Field(
        ..., description="Subset of problem cases chosen by the strategy"
    )
    mean_score: float = Field(..., description="Mean metric score across all cases")
    min_score: float = Field(..., description="Minimum metric score observed")
    max_score: float = Field(..., description="Maximum metric score observed")

    def to_reflection_dict(self, max_preview_chars: int = 200) -> dict[str, Any]:
        """
        Format the report as a plain dict suitable for
        ``ReflectionOptions(extra_feedback=report.to_reflection_dict())``.

        Prefer the ``dataset_report`` field on :class:`ReflectionOptions` for
        richer formatting in the prompt.
        """
        return {
            "dataset_evaluation_metric": self.metric_name,
            "dataset_evaluation_stats": (
                f"total={len(self.all_results)}, "
                f"mean={round(self.mean_score, 3)}, "
                f"min={round(self.min_score, 3)}, "
                f"max={round(self.max_score, 3)}"
            ),
            "dataset_problem_cases": "; ".join(
                f"[{r.case.label or f'case_{i + 1}'}] "
                f"score={round(r.score, 3)} "
                f"input={r.case.input[:max_preview_chars]!r} "
                f"output={r.chain_output[:max_preview_chars]!r}"
                for i, r in enumerate(self.selected_cases)
            )
            or "none",
            "dataset_overfitting_warning": (
                "Optimize for patterns visible across problem cases, "
                "not individual quirks. Mean score provides baseline context."
            ),
        }
