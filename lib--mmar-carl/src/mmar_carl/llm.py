"""
LLM client implementations for CARL using mmar-llm library.

Provides integration with mmar-llm library for production use.
"""

from mmar_llm import LLMHub
from mmar_mapi.services import LLMHubAPI, LLMCallProps
from .models import LLMClientBase


def create_llm_client(api, endpoint_key: str) -> LLMClientBase:
    """
    Factory function to create the appropriate LLM client based on the API type.

    Args:
        api: Either LLMHub, LLMHubAPI, or dynamically created client
        endpoint_key: Key for the specific entrypoint to use

    Returns:
        Appropriate LLM client instance

    Raises:
        ValueError: If api type is not supported or endpoint_key is empty
    """
    if not endpoint_key:
        raise ValueError("endpoint_key is required and cannot be empty")

    # Check for EntrypointsAccessor interface (supports __getitem__ for entrypoint access)
    if hasattr(api, "__getitem__"):
        return EntrypointsAccessorLLMClient(api, endpoint_key)

    # Check for LLMHubAPI interface (has get_response method)
    elif hasattr(api, "get_response"):
        return LLMAccessorClient(api, endpoint_key)

    # Check for mock classes by type name for testing
    api_type_name = type(api).__name__
    if "LLMHub" in api_type_name:
        return EntrypointsAccessorLLMClient(api, endpoint_key)
    elif "LLMHubAPI" in api_type_name or "PTAG" in api_type_name:
        return LLMAccessorClient(api, endpoint_key)

    # Fallback: if it has __getitem__, treat as entrypoint accessor
    elif hasattr(api, "__getitem__"):
        return EntrypointsAccessorLLMClient(api, endpoint_key)

    else:
        raise ValueError(
            f"Unsupported API type: {type(api).__name__}. "
            "Expected object with __getitem__ (EntrypointsAccessor) or "
            "get_response method (LLMHubAPI)"
        )


class EntrypointsAccessorLLMClient(LLMClientBase):
    """
    LLM client implementation using mmar-llm library.
    """

    def __init__(self, entrypoints: LLMHub, endpoint_key: str):
        if not endpoint_key:
            raise ValueError("endpoint_key is required and cannot be empty")

        self.entrypoints = entrypoints
        self.endpoint_key = endpoint_key

    async def get_response(self, prompt: str) -> str:
        ep = self.entrypoints[self.endpoint_key]
        result = ep.get_response_with_retries(prompt, retries=1)
        if hasattr(result, "__await__") or hasattr(result, "__aiter__"):
            return await result
        else:
            return result

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        ep = self.entrypoints[self.endpoint_key]
        result = ep.get_response_with_retries(prompt, retries=retries)
        if hasattr(result, "__await__") or hasattr(result, "__aiter__"):
            return await result
        else:
            return result


class LLMAccessorClient(LLMClientBase):
    """
    LLM client implementation using mmar-mapi library.
    """

    def __init__(self, api: LLMHubAPI, endpoint_key: str):
        if not endpoint_key:
            raise ValueError("endpoint_key is required and cannot be empty")

        self.api = api
        self.endpoint_key = endpoint_key

    async def get_response(self, prompt: str) -> str:
        props = LLMCallProps(endpoint_key=self.endpoint_key, attempts=1)
        result = self.api.get_response(request=prompt, props=props)
        if hasattr(result, "__await__") or hasattr(result, "__aiter__"):
            return await result
        else:
            return result

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        props = LLMCallProps(endpoint_key=self.endpoint_key, attempts=retries)
        result = self.api.get_response(request=prompt, props=props)
        # If the result is a coroutine (async), await it
        if hasattr(result, "__await__") or hasattr(result, "__aiter__"):
            return await result
        else:
            return result
