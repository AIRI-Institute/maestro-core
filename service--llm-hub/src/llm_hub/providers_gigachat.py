"""Gigachat native handler using gpt2giga library components."""

import json
import re
import uuid
from base64 import b64decode
from functools import partial
from typing import Any
from warnings import warn

from fastapi.responses import Response, StreamingResponse
from gigachat import GigaChat
from gigachat.exceptions import ResponseError
from gigachat.models import Chat, Function, FunctionParameters

# Import gpt2giga components
from gpt2giga.models.config import ProxyConfig, ProxySettings
from gpt2giga.protocol import (
    AttachmentProcessor,
    RequestTransformer,
    ResponseProcessor,
)
from loguru import logger

from llm_hub.config import ConnectionConfig
from llm_hub.providers_base import BaseProvider

# GigaChat authentication patterns
PATTERN_CID = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
CPATTERN_CID = re.compile(PATTERN_CID)


def _validate_cid(field_value: str, field: str = "field") -> None:
    """Validate CID format matches expected pattern."""
    if CPATTERN_CID.fullmatch(field_value):
        return
    warn(f"Maybe '{field}' is invalid: not matched by '{PATTERN_CID}'", stacklevel=2)


_validate_client_id = partial(_validate_cid, field="client_id")
_validate_client_secret = partial(_validate_cid, field="client_secret")


class GigachatProvider(BaseProvider):
    """Handler for native Gigachat API using gpt2giga library components."""

    def __init__(self, enable_images: bool = True):
        """Initialize the Gigachat handler.

        Args:
            enable_images: Whether to enable image upload support.
        """
        self.enable_images = enable_images
        self.clients: dict[str, GigaChat] = {}
        self.attachment_processor = AttachmentProcessor(logger)

    @staticmethod
    def _add_credentials(giga_params: dict, connection: ConnectionConfig) -> None:
        """Add credentials to params.

        Args:
            giga_params: Params dict to modify.
            connection: Connection configuration.
        """
        if connection.authorization:
            giga_params["credentials"] = connection.authorization
        elif connection.api_key:
            giga_params["credentials"] = connection.api_key

        # Handle user/password authentication
        if connection.user and connection.password:
            giga_params["user"] = connection.user
            giga_params["password"] = connection.password
            giga_params.pop("credentials", None)

    @staticmethod
    def _validate_authorization(authorization: str) -> None:
        """Validate authorization format.

        Args:
            authorization: Authorization string to validate.
        """
        try:
            authorization_decode = b64decode(authorization).decode()
            if ":" in authorization_decode:
                client_id, client_secret = authorization_decode.split(":", 1)
                _validate_client_id(client_id)
                _validate_client_secret(client_secret)
        except Exception:
            logger.debug("Authorization key in different format, skipping validation")

    @staticmethod
    def _add_optional_params(giga_params: dict, connection: ConnectionConfig) -> None:
        """Add optional GigaChat parameters.

        Args:
            giga_params: Params dict to modify.
            connection: Connection configuration.
        """
        if connection.scope:
            giga_params["scope"] = connection.scope
        if connection.auth_url:
            giga_params["auth_url"] = connection.auth_url
        if connection.access_token:
            giga_params["access_token"] = connection.access_token
        if connection.profanity_check is not None:
            giga_params["profanity_check"] = connection.profanity_check

        if connection.api_base and connection.api_base != "https://gigachat.devices.sberbank.ru/api/v1":
            giga_params["base_url"] = connection.api_base

    @staticmethod
    def _build_giga_params(connection: ConnectionConfig) -> dict:
        """Build GigaChat client parameters from connection config.

        Args:
            connection: Connection configuration.

        Returns:
            Dictionary of parameters for GigaChat constructor.
        """
        giga_params = {"verify_ssl_certs": connection.verify_ssl}
        GigachatProvider._add_credentials(giga_params, connection)

        if connection.authorization:
            GigachatProvider._validate_authorization(connection.authorization)

        GigachatProvider._add_optional_params(giga_params, connection)
        return giga_params

    def _get_client(self, connection: ConnectionConfig) -> GigaChat:
        """Get or create GigaChat client for connection.

        Args:
            connection: Connection configuration.

        Returns:
            GigaChat client instance.
        """
        # Create a unique key for this connection configuration
        key = (
            connection.api_base,
            connection.api_key,
            connection.authorization,
            connection.scope,
            connection.auth_url,
        )
        if key not in self.clients:
            giga_params = self._build_giga_params(connection)
            self.clients[key] = GigaChat(**giga_params)
        return self.clients[key]

    async def chat_completion(self, headers: dict[str, str], data: dict, connection: ConnectionConfig):
        """Handle chat completion request.

        Transforms OpenAI request to Gigachat format and processes response.
        """
        giga_client = self._get_client(connection)
        response_id = str(uuid.uuid4())

        # Create a simple config for the transformer
        proxy_settings = ProxySettings(enable_images=self.enable_images)
        config = ProxyConfig(proxy=proxy_settings)

        transformer = RequestTransformer(
            config,
            logger,
            self.attachment_processor,
        )
        response_processor = ResponseProcessor(logger)

        try:
            # Transform request to Gigachat format
            transformed = await transformer.prepare_chat_completion(data, giga_client)

            # Convert to Chat model
            messages = transformed.get("messages", [])
            functions = transformed.get("functions")
            function_call = transformed.get("function_call")
            model = transformed.get("model")
            temperature = transformed.get("temperature")
            max_tokens = transformed.get("max_tokens")
            top_p = transformed.get("top_p")

            # Convert functions to GigaChat format
            giga_functions = None
            if functions:
                giga_functions = [
                    Function(
                        name=f.get("name"),
                        description=f.get("description", ""),
                        parameters=FunctionParameters(**f.get("parameters", {})),
                    )
                    for f in functions
                ]

            chat = Chat(
                messages=messages,
                functions=giga_functions,
                function_call=function_call,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            if data.get("stream", False):
                # Streaming response
                return StreamingResponse(
                    self._stream_generator(data, giga_client, chat, response_id, response_processor),
                    media_type="text/event-stream",
                )
            else:
                # Non-streaming response
                giga_response = await giga_client.achat(chat)
                return response_processor.process_response(
                    giga_response, data.get("model", "gigachat"), response_id, request_data=data
                )

        except ResponseError as e:
            logger.error(f"GigaChat response error in chat_completion: {e}")
            return Response(
                content=json.dumps({"error": {"message": str(e), "type": "api_error"}}),
                status_code=500,
                media_type="application/json",
            )
        except Exception as e:
            logger.exception(f"Error in chat_completion: {e}")
            return Response(
                content=json.dumps({"error": {"message": str(e), "type": "internal_error"}}),
                status_code=500,
                media_type="application/json",
            )

    async def _stream_generator(
        self,
        data: dict[str, Any],
        giga_client: GigaChat,
        chat: Chat,
        response_id: str,
        response_processor: ResponseProcessor,
    ):
        """Generator for streaming responses."""
        try:
            async for chunk in giga_client.astream(chat):
                processed = response_processor.process_stream_chunk(
                    chunk, data.get("model", "gigachat"), response_id, request_data=data
                )
                yield f"data: {json.dumps(processed)}\n\n"
        except ResponseError as e:
            logger.error(f"GigaChat response error in stream generator: {e}")
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": "internal_error",
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"
        except Exception as e:
            logger.exception(f"Error in stream generator: {e}")
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "stream_error",
                    "code": "internal_error",
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    async def embeddings(self, headers: dict[str, str], data: dict, connection: ConnectionConfig):
        """Handle embeddings request for Gigachat.

        Args:
            headers: HTTP headers (not used for Gigachat).
            data: Request body as dictionary.
            connection: Connection configuration.

        Returns:
            Response in OpenAI embeddings format.
        """
        giga_client = self._get_client(connection)

        try:
            inputs = data.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]

            # Call Gigachat embeddings API
            embeddings_response = await giga_client.aembeddings(
                texts=inputs,
                model=connection.embeddings_model,
            )

            # Transform to OpenAI format
            total_prompt_tokens = 0
            for emb in embeddings_response.data or []:
                if hasattr(emb, "usage") and emb.usage:
                    total_prompt_tokens += getattr(emb.usage, "prompt_tokens", 0)

            return {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": emb.embedding,
                        "index": i,
                    }
                    for i, emb in enumerate(embeddings_response.data or [])
                ],
                "model": data.get("model", "gigachat"),
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "total_tokens": total_prompt_tokens,
                },
            }

        except ResponseError as e:
            logger.error(f"GigaChat response error in embeddings: {e}")
            return Response(
                content=json.dumps({"error": {"message": str(e), "type": "api_error"}}),
                status_code=500,
                media_type="application/json",
            )
        except Exception as e:
            logger.exception(f"Error in embeddings: {e}")
            return Response(
                content=json.dumps({"error": {"message": str(e), "type": "internal_error"}}),
                status_code=500,
                media_type="application/json",
            )
