"""
Abstract base classes for CARL reasoning system.
"""

from abc import ABC, abstractmethod


class LLMClientBase(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def get_response(self, prompt: str) -> str:
        """
        Get a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM response as a string
        """
        pass

    @abstractmethod
    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """
        Get a response from the LLM with retry logic.

        Args:
            prompt: The prompt to send to the LLM
            retries: Maximum number of retry attempts

        Returns:
            The LLM response as a string
        """
        pass
