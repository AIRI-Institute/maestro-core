"""
Abstract base class for step and chain metrics in CARL.

Usage example:
    ```python
    from mmar_carl import MetricBase, MetricOutput
    from mmar_carl.models import StepExecutionResult, ReasoningResult

    class WordCountMetric(MetricBase):
        @property
        def name(self) -> str:
            return "word_count"

        async def compute_async(self, output: MetricOutput) -> float:
            if isinstance(output, ReasoningResult):
                text = output.get_final_output()
            else:
                text = output.result
            return float(len(text.split()))

    class LLMJudgeMetric(MetricBase):
        def __init__(self, api, model: str):
            self._api = api
            self._model = model

        @property
        def name(self) -> str:
            return "llm_judge_score"

        async def compute_async(self, output: MetricOutput) -> float:
            # For chain-level metrics, output is ReasoningResult — access full history,
            # step results, etc.  For step-level metrics, output is StepExecutionResult.
            if isinstance(output, ReasoningResult):
                text = output.get_final_output()
            else:
                text = output.result
            # Call LLM to score the text, parse the numeric result
            ...
            return score

    # Attach to a step
    step = LLMStepDescription(
        number=1,
        title="Analysis",
        aim="Analyze the data",
        metrics=[WordCountMetric()],
    )

    # Attach to a chain
    chain = ReasoningChain(steps=[step], metrics=[LLMJudgeMetric(api, "gpt-4")])
    ```
"""

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .models.results import ReasoningResult, StepExecutionResult

#: Union type for the ``output`` parameter of :meth:`MetricBase.compute_async`.
#: At step level the value is a :class:`~mmar_carl.models.results.StepExecutionResult`;
#: at chain level it is a :class:`~mmar_carl.models.results.ReasoningResult`.
MetricOutput = Union["StepExecutionResult", "ReasoningResult"]


class MetricBase(ABC):
    """
    Abstract base class for evaluation metrics.

    Implement this class to attach custom evaluation logic to steps or chains.
    The metric receives the full execution output and returns a numeric score.

    The ``output`` parameter is a
    :class:`~mmar_carl.models.results.StepExecutionResult` when the metric is
    attached to a step, and a :class:`~mmar_carl.models.results.ReasoningResult`
    when attached to a chain.  Use ``isinstance`` to branch if your metric needs
    to handle both levels:

    .. code-block:: python

        async def compute_async(self, output: MetricOutput) -> float:
            if isinstance(output, ReasoningResult):
                text = output.get_final_output()
            else:               # StepExecutionResult
                text = output.result
            return float(len(text.split()))

    The score can represent anything: word count, LLM-as-a-judge rating,
    file-parsed value, cosine similarity, etc.

    **Reflection integration**: when ``ReflectionOptions.include_metric_scores=True``
    (the default), all computed scores are automatically included in the reflection
    prompt. The LLM can then reference them as concrete quality signals — for example,
    a low ``keyword_coverage`` score on a step will prompt it to suggest better
    ``step_context_queries``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique metric name.

        Used as the key in the ``metrics`` dict of execution results.
        Must be unique among metrics attached to the same step or chain.
        """
        ...

    @abstractmethod
    async def compute_async(self, output: MetricOutput) -> float:
        """
        Compute the metric value from a step or chain output.

        Args:
            output: The execution result to evaluate.  This is a
                :class:`~mmar_carl.models.results.StepExecutionResult` for
                step-level metrics and a
                :class:`~mmar_carl.models.results.ReasoningResult` for
                chain-level metrics.

        Returns:
            A numeric score (float).
        """
        ...

    def compute(self, output: MetricOutput) -> float:
        """
        Synchronous wrapper around :meth:`compute_async`.

        Prefer :meth:`compute_async` in async contexts.
        """
        return asyncio.run(self.compute_async(output))
