"""Anthropic API provider using official anthropic library."""

import json
import logging

import anthropic
from fastapi.responses import Response, StreamingResponse

from llm_hub.config import ConnectionConfig
from llm_hub.providers_base import BaseProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic API using official anthropic library.

    Converts OpenAI-compatible requests to Anthropic's native format
    and converts responses back to OpenAI format.
    """

    def __init__(self):
        """Initialize the Anthropic provider."""
        self.clients: dict[str, anthropic.Anthropic] = {}

    def _get_client(self, connection: ConnectionConfig) -> anthropic.Anthropic:
        """Get or create Anthropic client for connection.

        Args:
            connection: Connection configuration.

        Returns:
            Anthropic client instance.
        """
        # Create a unique key for this connection configuration
        key = (connection.api_base, connection.api_key)

        if key not in self.clients:
            self.clients[key] = anthropic.Anthropic(
                api_key=connection.api_key,
                base_url=connection.api_base,
            )

        return self.clients[key]

    def _convert_to_openai_format(self, message: anthropic.types.Message) -> dict:
        """Convert Anthropic Message to OpenAI format.

        Args:
            message: Anthropic Message object

        Returns:
            OpenAI-style response dict
        """
        # Extract text content from content blocks
        text_content = ""
        for block in message.content:
            if block.type == "text":
                text_content += block.text

        return {
            "id": message.id,
            "object": "chat.completion",
            "created": 0,  # Anthropic doesn't provide timestamp
            "model": message.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text_content,
                    },
                    "finish_reason": message.stop_reason,
                }
            ],
            "usage": {
                "prompt_tokens": message.usage.input_tokens,
                "completion_tokens": message.usage.output_tokens,
                "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
            },
        }

    async def chat_completion(self, headers: dict[str, str], data: dict, connection: ConnectionConfig):
        """Handle chat completion request.

        Converts OpenAI request to Anthropic native format, forwards it,
        and converts response back to OpenAI format.
        """
        client = self._get_client(connection)

        # Extract system message if present
        system_content = None
        anthropic_messages = []

        for msg in data.get("messages", []):
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_content = content
            else:
                anthropic_messages.append({"role": role, "content": content})

        # Build Anthropic request parameters
        params = {
            "model": data.get("model", "claude-sonnet-4-5-20250929"),
            "messages": anthropic_messages,
            "max_tokens": data.get("max_tokens", 1024),
        }

        if system_content:
            params["system"] = system_content

        if "temperature" in data:
            params["temperature"] = data["temperature"]

        if "top_p" in data:
            params["top_p"] = data["top_p"]

        logger.debug(f"Anthropic request params: {params}")

        try:
            if data.get("stream", False):
                # Streaming response
                async def stream_generator():
                    try:
                        with client.messages.stream(**params) as stream:
                            for event in stream:
                                # Convert Anthropic streaming event to OpenAI format
                                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                                    openai_chunk = {
                                        "id": stream.current_message_snapshot.id if hasattr(stream, "current_message_snapshot") else "",
                                        "object": "chat.completion.chunk",
                                        "created": 0,
                                        "model": params["model"],
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": event.delta.text},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                    yield f"data: {json.dumps(openai_chunk)}\n\n"

                            # Send final chunk with finish_reason
                            if hasattr(stream, "current_message_snapshot"):
                                final_chunk = {
                                    "id": stream.current_message_snapshot.id,
                                    "object": "chat.completion.chunk",
                                    "created": 0,
                                    "model": params["model"],
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": stream.current_message_snapshot.stop_reason,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(final_chunk)}\n\n"

                        yield "data: [DONE]\n\n"

                    except anthropic.APIError as e:
                        logger.error(f"Anthropic API error in streaming: {e}")
                        error_response = {
                            "error": {
                                "message": str(e),
                                "type": "api_error",
                            }
                        }
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"

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
                message = client.messages.create(**params)

                # Convert Anthropic response to OpenAI format
                openai_response = self._convert_to_openai_format(message)

                return Response(
                    content=json.dumps(openai_response),
                    status_code=200,
                    headers={"Content-Type": "application/json"},
                )

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "api_error",
                }
            }
            return Response(
                content=json.dumps(error_response),
                status_code=500,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            logger.exception(f"Unexpected error in Anthropic provider: {e}")
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                }
            }
            return Response(
                content=json.dumps(error_response),
                status_code=500,
                headers={"Content-Type": "application/json"},
            )

    async def embeddings(self, headers: dict[str, str], data: dict, connection: ConnectionConfig):
        """Handle embeddings request.

        Note: Anthropic does not have a native embeddings endpoint.
        This is included for interface compatibility but will return an error.
        """
        logger.warning("Anthropic does not support embeddings endpoint")
        return Response(
            content='{"error": {"message": "Anthropic does not support embeddings", "type": "not_supported"}}',
            status_code=400,
            headers={"Content-Type": "application/json"},
        )
