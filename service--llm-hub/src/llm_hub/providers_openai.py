"""OpenAI-compatible API pass-through handler."""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi.responses import Response, StreamingResponse

from llm_hub.config import ConnectionConfig
from llm_hub.providers_base import BaseProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI-compatible APIs (pass-through)."""

    def __init__(self, timeout: float = 120.0):
        """Initialize the OpenAI provider.

        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout

    @asynccontextmanager
    async def _get_client(self):
        """Get an HTTP client context manager."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            yield client

    async def chat_completion(self, headers: dict[str, str], data: dict, connection: ConnectionConfig):
        """Handle chat completion request.

        Forwards the request to the upstream OpenAI-compatible API.
        """
        request_headers = {
            "Authorization": f"Bearer {connection.api_key}",
            "Content-Type": "application/json",
        }

        # Forward original headers that might be relevant
        forwardable_keys = {"x-request-id", "x-customer-id", "user-agent"}
        for key, value in headers.items():
            if key.lower() in forwardable_keys:
                request_headers[key] = value

        url = f"{connection.api_base.rstrip('/')}/chat/completions"

        if data.get("stream", False):
            # Streaming response - use generator to keep client alive
            async def stream_generator():
                async with (
                    httpx.AsyncClient(timeout=self.timeout) as client,
                    client.stream("POST", url, json=data, headers=request_headers) as response,
                ):
                    if response.status_code >= 400:
                        error_content = await response.aread()
                        logger.error(f"Upstream error {response.status_code}: {error_content}")
                        yield f"data: {error_content.decode()}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    async for chunk in response.aiter_bytes():
                        yield chunk

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Non-streaming response
            async with self._get_client() as client:
                response = await client.post(
                    url,
                    json=data,
                    headers=request_headers,
                )

                if response.status_code >= 400:
                    logger.error(f"Upstream error {response.status_code}: {response.text}")

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers={"Content-Type": "application/json"},
                )

    async def embeddings(self, headers: dict[str, str], data: dict, connection: ConnectionConfig):
        """Handle embeddings request.

        Forwards the request to the upstream OpenAI-compatible API.
        """
        request_headers = {
            "Authorization": f"Bearer {connection.api_key}",
            "Content-Type": "application/json",
        }

        # Forward original headers that might be relevant
        forwardable_keys = {"x-request-id", "x-customer-id", "user-agent"}
        for key, value in headers.items():
            if key.lower() in forwardable_keys:
                request_headers[key] = value

        url = f"{connection.api_base.rstrip('/')}/embeddings"

        async with self._get_client() as client:
            response = await client.post(
                url,
                json=data,
                headers=request_headers,
            )

            if response.status_code >= 400:
                logger.error(f"Upstream error {response.status_code}: {response.text}")

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers={"Content-Type": "application/json"},
            )
