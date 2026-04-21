"""
LLM client implementations for CARL.

Provides integration with:
- OpenAI-compatible APIs (OpenRouter, Azure OpenAI, local LLMs, etc.)
"""

import asyncio
import os
from typing import Any, Callable, Optional

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pydantic import BaseModel, Field

from mmar_carl.models import LLMClientBase

# Environment variable for OpenAI-compatible API base URL
# Defaults to OpenRouter if not set
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


class OpenAIClientConfig(BaseModel):
    """Configuration for OpenAI-compatible LLM clients."""

    base_url: str = Field(default_factory=lambda: OPENAI_BASE_URL, description="Base URL for the OpenAI-compatible API")
    api_key: str = Field(..., description="API key for authentication")
    model: str = Field(..., description="Model identifier (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet')")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens in response (None for model default)")
    timeout: float = Field(default=120.0, gt=0, description="Request timeout in seconds")
    verify_ssl: bool = Field(default=True, description="Whether to verify TLS certificates for HTTPS connections")
    extra_headers: dict[str, str] = Field(default_factory=dict, description="Additional HTTP headers")
    extra_body: dict[str, Any] = Field(default_factory=dict, description="Additional request body parameters")


class OpenAICompatibleClient(LLMClientBase):
    """
    LLM client for OpenAI-compatible APIs (OpenRouter, Azure OpenAI, local LLMs, etc.).

    This client uses the official OpenAI Python library to communicate with any
    OpenAI-compatible API. It supports:
    - OpenRouter (default)
    - Azure OpenAI
    - Local LLMs with OpenAI-compatible APIs (LM Studio, Ollama, vLLM, etc.)
    - Any other OpenAI-compatible service

    Example usage with OpenRouter:
        ```python
        config = OpenAIClientConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-v1-...",
            model="anthropic/claude-3.5-sonnet",
            extra_headers={
                "HTTP-Referer": "https://your-site.com",
                "X-Title": "Your App Name"
            }
        )
        client = OpenAICompatibleClient(config)
        response = await client.get_response("Hello!")
        ```

    Example usage with local LLM:
        ```python
        config = OpenAIClientConfig(
            base_url="http://localhost:1234/v1",
            api_key="not-needed",
            model="local-model",
        )
        client = OpenAICompatibleClient(config)
        ```
    """

    def __init__(self, config: OpenAIClientConfig):
        """
        Initialize the OpenAI-compatible client.

        Args:
            config: Configuration for the client

        Raises:
            ImportError: If openai package is not installed
        """
        self.config = config
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> "AsyncOpenAI":
        """Lazy initialization of the AsyncOpenAI client."""
        if self._client is None:
            client_kwargs: dict[str, Any] = {
                "base_url": self.config.base_url,
                "api_key": self.config.api_key,
                "timeout": self.config.timeout,
                "default_headers": self.config.extra_headers if self.config.extra_headers else None,
            }

            # Some enterprise/self-hosted gateways use custom certificates.
            # Allow opting out of TLS verification for compatibility.
            if not self.config.verify_ssl:
                if DefaultAsyncHttpxClient is not None:
                    client_kwargs["http_client"] = DefaultAsyncHttpxClient(
                        verify=False,
                        timeout=self.config.timeout,
                    )
                else:
                    client_kwargs["http_client"] = httpx.AsyncClient(
                        verify=False,
                        timeout=self.config.timeout,
                    )

            self._client = AsyncOpenAI(
                **client_kwargs,
            )
        return self._client

    async def get_response(self, prompt: str) -> str:
        """
        Get a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM response as a string
        """
        return await self._make_request(prompt)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """
        Get a response from the LLM with retry logic.

        Args:
            prompt: The prompt to send to the LLM
            retries: Maximum number of retry attempts

        Returns:
            The LLM response as a string

        Raises:
            Exception: If all retries fail
        """
        last_error: Optional[Exception] = None

        for attempt in range(retries):
            try:
                return await self._make_request(prompt)
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    # Exponential backoff: 1s, 2s, 4s, ...
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)

        raise last_error or Exception("All retries failed")

    async def _make_request(self, prompt: str) -> str:
        """
        Make a single request to the LLM.

        Args:
            prompt: The prompt to send

        Returns:
            The response content as a string
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
        }

        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens

        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body

        response = await self.client.chat.completions.create(**kwargs)

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""

    async def stream_response(self, prompt: str) -> Any:
        """
        Stream a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM

        Yields:
            Chunks of the response as they arrive

        Example:
            ```python
            async for chunk in client.stream_response("Hello!"):
                print(chunk, end="", flush=True)
            ```
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "stream": True,
        }

        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens

        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body

        async with await self.client.chat.completions.create(**kwargs) as stream:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    async def get_response_streaming(self, prompt: str, on_chunk: Optional[Callable[[str], None]] = None) -> str:
        """
        Get a response from the LLM with optional streaming callback.

        Args:
            prompt: The prompt to send to the LLM
            on_chunk: Optional callback called with each chunk

        Returns:
            The complete LLM response as a string
        """
        full_response = ""
        async for chunk in self.stream_response(prompt):
            full_response += chunk
            if on_chunk:
                on_chunk(chunk)
        return full_response

    async def close(self) -> None:
        """
        Close the underlying OpenAI client and release resources.

        This should be called when the client is no longer needed to ensure
        proper cleanup of httpx connections and avoid event loop issues.
        """
        if self._client is not None:
            await self._client.close()
            self._client = None


def create_openai_client(
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: float = 120.0,
    verify_ssl: bool = True,
    extra_headers: Optional[dict[str, str]] = None,
    extra_body: Optional[dict[str, Any]] = None,
) -> OpenAICompatibleClient:
    """
    Factory function to create an OpenAI-compatible LLM client.

    This is a convenience function for creating OpenAICompatibleClient instances.

    Args:
        api_key: API key for authentication
        model: Model identifier (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet')
        base_url: Base URL for the API (default: from OPENAI_BASE_URL env var,
            or OpenRouter if not set)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response (None for model default)
        timeout: Request timeout in seconds (default: 120.0)
        verify_ssl: Whether to verify TLS certificates (default: True)
        extra_headers: Additional HTTP headers
        extra_body: Additional request body parameters

    Returns:
        Configured OpenAICompatibleClient instance

    Example:
        ```python
        # Using env var (OPENAI_BASE_URL) or default (OpenRouter)
        client = create_openai_client(
            api_key="sk-or-v1-...",
            model="anthropic/claude-3.5-sonnet"
        )

        # Explicit base_url overrides env var
        client = create_openai_client(
            api_key="not-needed",
            model="llama3",
            base_url="http://localhost:11434/v1"
        )
        ```
    """
    config = OpenAIClientConfig(
        base_url=base_url or OPENAI_BASE_URL,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        verify_ssl=verify_ssl,
        extra_headers=extra_headers or {},
        extra_body=extra_body or {},
    )
    return OpenAICompatibleClient(config)
