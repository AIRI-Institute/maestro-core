"""LLM Hub OpenAI client that mimics the OpenAI API structure."""

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import wraps
from types import SimpleNamespace as SNS
from typing import Any

from fastapi import HTTPException
from fastapi.responses import Response
from loguru import logger

from llm_hub.config import ConnectionConfig, LLMConfig
from llm_hub.providers_anthropic import AnthropicProvider
from llm_hub.providers_base import BaseProvider
from llm_hub.providers_gigachat import GigachatProvider
from llm_hub.providers_openai import OpenAIProvider

# Provider registry: maps api_type to provider class
PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "gigachat": GigachatProvider,
    "anthropic": AnthropicProvider,
}


def _with_semaphore(
    semaphores: dict[str, asyncio.Semaphore],
    get_model_id: Callable[..., str],
) -> Callable:
    """Wrap an async method to acquire/release semaphore based on model.

    Args:
        semaphores: Dictionary mapping model_id to Semaphore.
        get_model_id: Function to extract model_id from call arguments.

    Returns:
        Decorator function that wraps the method.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            model_id = get_model_id(*args, **kwargs)
            semaphore = semaphores.get(model_id)
            if semaphore:
                await semaphore.acquire()
                try:
                    return await func(*args, **kwargs)
                finally:
                    semaphore.release()
            else:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def _create_model_data(model_id: str, model_info: Any | None = None, **extra) -> dict:
    """Create a model data dictionary.

    Args:
        model_id: Model identifier.
        model_info: Optional ModelInfo object.
        **extra: Extra fields to add.

    Returns:
        Model data dictionary.
    """
    data = {
        "id": model_id,
        "object": "model",
        "created": 1677610602,
        "owned_by": model_info.owned_by if model_info else "unknown",
    }
    if model_info and model_info.caption:
        data["caption"] = model_info.caption
    data.update(extra)
    return data


class OpenAIProto:
    pass


# todo what is base class?
class LLMHub(OpenAIProto):
    """LLM Hub OpenAI client that mimics the OpenAI API structure.

    This class provides a drop-in replacement for the OpenAI client with the
    same method signatures: `.chat.completions.create()` and `.embeddings.create()`.

    Example:
        client = LLMHub(config)

        # Chat completions
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Embeddings
        response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input="Hello world",
        )
    """

    def __init__(self, config: LLMConfig):
        """Initialize the LLM Hub OpenAI client.

        Args:
            config: Application configuration.
        """
        self._config = config
        self._providers: dict[str, BaseProvider] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._semaphore_lock = asyncio.Lock()

        # Initialize provider instances
        for api_type, provider_class in PROVIDER_CLASSES.items():
            self._providers[api_type] = provider_class()

        # Nested API objects (mimics OpenAI structure)
        self.chat = SNS(completions=SNS(create=self._chat_completion_create))
        self.embeddings = SNS(create=self._embeddings_create)
        self.models = SNS(list=self._build_models_list)

    def _create_semaphore_for_model(self, model: str) -> asyncio.Semaphore | None:
        """Create semaphore for model if configured.

        Returns semaphore or None if no limit set.
        """
        config = self._config

        # First check model_info for explicit model-level limits
        if model in config.model_info:
            model_info = config.model_info[model]
            if model_info.max_concurrent is not None and model_info.max_concurrent > 0:
                logger.info(
                    f"Created semaphore for model '{model}' with max_concurrent={model_info.max_concurrent} (from model_info)"
                )
                return asyncio.Semaphore(model_info.max_concurrent)

        # Then check connection config for connection-level limits
        if model in config.routing:
            route = config.routing[model]
            if "." in route:
                conn_name, _ = route.split(".", 1)
                if conn_name in config.connections:
                    connection = config.connections[conn_name]
                    if connection.max_concurrent is not None and connection.max_concurrent > 0:
                        logger.info(
                            f"Created semaphore for model '{model}' with max_concurrent={connection.max_concurrent} (from connection '{conn_name}')"
                        )
                        return asyncio.Semaphore(connection.max_concurrent)

        return None

    @asynccontextmanager
    async def _get_semaphore(self, model: str):
        """Acquire semaphore for model if configured (lazy creation)."""
        # Fast path: semaphore already exists
        semaphore = self._semaphores.get(model)
        if semaphore:
            await semaphore.acquire()
            try:
                yield
            finally:
                semaphore.release()
            return

        # Slow path: create semaphore if needed
        async with self._semaphore_lock:
            # Check again in case another thread created it
            semaphore = self._semaphores.get(model)
            if semaphore:
                await semaphore.acquire()
                try:
                    yield
                finally:
                    semaphore.release()
                return

            # Create and cache semaphore
            semaphore = self._create_semaphore_for_model(model)
            if semaphore:
                self._semaphores[model] = semaphore
                await semaphore.acquire()
                try:
                    yield
                finally:
                    semaphore.release()
            else:
                yield

    async def _chat_completion_create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> Response | dict[str, Any]:
        """Create a chat completion.

        Args:
            model: Model name to use.
            messages: List of message dictionaries.
            headers: Optional HTTP headers to forward (e.g., x-request-id).
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            Response from the appropriate provider.

        Raises:
            HTTPException: If model or connection not found, or request fails.
        """
        model = self._fix_model_if_needed(model, is_embedding=False)
        connection, provider, resolved_model = self._resolve_connection(model, is_embedding=False)

        async with self._get_semaphore(model):
            return await self._do_chat_completion(provider, connection, resolved_model, headers, messages, kwargs)

    async def _do_chat_completion(
        self,
        provider: BaseProvider,
        connection: ConnectionConfig,
        resolved_model: str,
        headers: dict[str, str] | None,
        messages: list[dict[str, Any]],
        kwargs: dict,
    ) -> Response | dict[str, Any]:
        """Execute chat completion request."""
        data = {
            "model": self._get_target_model(resolved_model),
            "messages": messages,
            **kwargs,
        }

        try:
            result = await provider.chat_completion(headers or {}, data, connection)
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error in chat_completion")
            raise HTTPException(
                status_code=500,
                detail={"error": {"message": str(e), "type": "internal_error"}},
            ) from None

    async def _embeddings_create(
        self,
        *,
        model: str,
        input: str | list[str] | list[int] | list[list[int]],  # noqa: A002
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> Response | dict[str, Any]:
        """Create an embedding.

        Args:
            model: Model name to use.
            input: Input text(s) to embed.
            headers: Optional HTTP headers to forward (e.g., x-request-id).
            **kwargs: Additional parameters.

        Returns:
            Response from the appropriate provider.

        Raises:
            HTTPException: If model or connection not found, or request fails.
        """
        model = self._fix_model_if_needed(model, is_embedding=True)
        connection, provider, resolved_model = self._resolve_connection(model, is_embedding=True)

        async with self._get_semaphore(model):
            return await self._do_embeddings(provider, connection, resolved_model, headers, input, kwargs, model)

    async def _do_embeddings(
        self,
        provider: BaseProvider,
        connection: ConnectionConfig,
        resolved_model: str,
        headers: dict[str, str] | None,
        input: str | list[str] | list[int] | list[list[int]],  # noqa: A002
        kwargs: dict,
        original_model: str,
    ) -> Response | dict[str, Any]:
        """Execute embeddings request."""
        data = {
            "model": self._get_target_model(resolved_model),
            "input": input,
            **kwargs,
        }

        try:
            result = await provider.embeddings(headers or {}, data, connection)
            logger.trace(f"Embeddings: done, model={original_model}")
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Embeddings: failed")
            raise HTTPException(
                status_code=500,
                detail={"error": {"message": str(e), "type": "internal_error"}},
            ) from None

    def _get_default_model(self, *, is_embedding: bool = False) -> str | None:
        """Get default model from config.

        Args:
            is_embedding: True if this is an embedding request.

        Returns:
            Default model ID or None if not configured.
        """
        config = self._config
        for model_id, info in config.model_info.items():
            if (is_embedding and info.default_embeddings and model_id in config.routing) or (
                not is_embedding and info.default and model_id in config.routing
            ):
                return model_id
        return None

    def _fix_model_if_needed(self, model: str, *, is_embedding: bool = False) -> str:
        """Fix model name if empty or not found in routing.

        Args:
            model: Model name from request.
            is_embedding: True if this is an embedding request.

        Returns:
            Fixed model name (either original, default, or fallback).

        Raises:
            HTTPException: If no default model configured or model not found.
        """
        config = self._config

        # Empty string: use default
        if not model:
            default = self._get_default_model(is_embedding=is_embedding)
            if default:
                return default
            raise HTTPException(status_code=400, detail="No default model configured")

        # Non-existent model: fallback to default or raise error
        if model not in config.routing:
            if config.fallback_to_default:
                logger.warning(f"Requested model '{model}' does not exist, falling back to default")
                default = self._get_default_model(is_embedding=is_embedding)
                if default:
                    return default
                raise HTTPException(status_code=400, detail=f"Unknown model: {model} and no default model configured")
            else:
                logger.error(f"Requested model '{model}' does not exist and fallback is disabled")
                raise HTTPException(status_code=404, detail=f"Unknown model: {model}")

        return model

    def _resolve_connection(
        self,
        model: str,
        *,
        is_embedding: bool = False,
    ) -> tuple[ConnectionConfig, BaseProvider, str]:
        """Resolve model name to connection config and provider.

        Args:
            model: Model name from request (already fixed/validated).
            is_embedding: True if this is an embedding request.

        Returns:
            Tuple of (connection_config, provider, resolved_model).

        Raises:
            HTTPException: If model or connection not found.
        """
        config = self._config
        providers = self._providers

        route = config.routing[model]
        if "." not in route:
            raise HTTPException(status_code=400, detail=f"Invalid route format: {route}")

        conn_name, _target_model = route.split(".", 1)
        if conn_name not in config.connections:
            raise HTTPException(status_code=400, detail=f"Unknown connection: {conn_name}")

        connection = config.connections[conn_name]
        api_type = connection.api_type

        if api_type not in providers:
            raise HTTPException(status_code=400, detail=f"Unsupported api_type: {api_type}")

        provider = providers[api_type]
        return connection, provider, model

    def _get_target_model(self, model: str) -> str:
        """Get the target model name from routing configuration.

        Args:
            model: Model name from request.

        Returns:
            Target model name to use in the upstream request.
        """
        route = self._config.routing[model]
        _, target_model = route.split(".", 1)
        return target_model

    def _get_or_create_model(self, models: list[dict], model_id: str, model_info: Any | None = None) -> dict:
        """Get existing model or create and add new one.

        Args:
            models: List of model dicts to search.
            model_id: Model ID to find or create.
            model_info: Optional ModelInfo object.

        Returns:
            The model dict (existing or newly created).
        """
        for model in models:
            if model["id"] == model_id:
                return model
        # Model not found, create and add
        model_data = _create_model_data(model_id, model_info)
        models.append(model_data)
        return model_data

    def _mark_defaults_from_info(self, models: list[dict], flag: str) -> None:
        """Mark models with default flags from model_info.

        Args:
            models: List of model dicts.
            flag: Either "default" or "default_embeddings".
        """
        config = self._config
        info_models = set()

        for model_id, info in config.model_info.items():
            if getattr(info, flag, False) and model_id in config.routing:
                info_models.add(model_id)

        if len(info_models) > 1:
            logger.warning(f"Multiple models marked as {flag} in model_info: {info_models}. Using first.")

        for model_id in info_models:
            model = self._get_or_create_model(models, model_id, config.model_info[model_id])
            model[flag] = True

    def _build_models_list(self) -> dict:
        """Build models list from configuration.

        Returns:
            Dictionary with "object" and "data" keys compatible with OpenAI API.
        """
        config = self._config
        models = []

        # Build models list from routing
        for model_id in config.routing:
            if model_id == "":
                continue
            model_info = config.model_info.get(model_id)
            models.append(_create_model_data(model_id, model_info))
        self._mark_defaults_from_info(models, "default")
        self._mark_defaults_from_info(models, "default_embeddings")

        return {
            "object": "list",
            "data": models,
        }
