"""
Abstract base classes for CARL reasoning system.
"""

from abc import ABC, abstractmethod
from typing import Any, List

from pydantic import BaseModel, Field

from mmar_carl.models.llm_client_base import LLMClientBase


class SelfCriticDecision(BaseModel):
    """
    Decision produced by a self-critic evaluator.

    Contract:
      - verdict: "APPROVE" or "DISAPPROVE"
      - review_text: human-readable review details
    """

    verdict: str
    review_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def normalized_verdict(self) -> str:
        """Normalize verdict into APPROVE/DISAPPROVE."""
        verdict = (self.verdict or "").strip().upper()
        if verdict == "APPROVE":
            return "APPROVE"
        return "DISAPPROVE"


class SelfCriticEvaluatorBase(ABC):
    """Abstract self-critic evaluator strategy."""

    @abstractmethod
    async def evaluate(
        self,
        step: Any,
        candidate: str,
        base_prompt: str,
        context: Any,
        llm_client: LLMClientBase,
        retries: int,
    ) -> SelfCriticDecision:
        """
        Evaluate candidate response and return approval decision.

        Args:
            step: Current step object
            candidate: Candidate response to evaluate
            base_prompt: Original step prompt used for generation
            context: Active reasoning context
            llm_client: LLM client for optional evaluator usage
            retries: Retry budget for evaluator model calls
        """
        pass


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    @abstractmethod
    def extract_context(self, outer_context: str, queries: List[str], **kwargs) -> str:
        """
        Extract relevant context from outer_context using queries.

        Args:
            outer_context: The full context data to search through
            queries: List of queries to find relevant context
            **kwargs: Additional strategy-specific parameters

        Returns:
            String containing relevant context found for each query
        """
        pass
