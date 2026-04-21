"""Base provider interface for connection types."""

from abc import ABC, abstractmethod

from llm_hub.config import ConnectionConfig


class BaseProvider(ABC):
    """Base interface for LLM providers."""

    @abstractmethod
    async def chat_completion(
        self,
        headers: dict[str, str],
        data: dict,
        connection: ConnectionConfig,
    ):
        """Handle chat completion request.

        Args:
            headers: HTTP headers to forward (e.g., x-request-id).
            data: Request body as dictionary.
            connection: Connection configuration.

        Returns:
            Response compatible with OpenAI API format.
        """
        pass

    @abstractmethod
    async def embeddings(
        self,
        headers: dict[str, str],
        data: dict,
        connection: ConnectionConfig,
    ):
        """Handle embeddings request.

        Args:
            headers: HTTP headers to forward (e.g., x-request-id).
            data: Request body as dictionary.
            connection: Connection configuration.

        Returns:
            Response compatible with OpenAI API format.
        """
        pass
