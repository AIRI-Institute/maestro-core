"""FastAPI utility functions."""

from fastapi import Request


def extract_forwardable_headers(request: Request) -> dict[str, str]:
    """Extract headers from request that should be forwarded to upstream.

    Args:
        request: FastAPI Request object.

    Returns:
        Dictionary of headers to forward.
    """
    forwardable_keys = {"x-request-id", "x-customer-id", "user-agent"}
    return {key: value for key, value in request.headers.items() if key.lower() in forwardable_keys}
