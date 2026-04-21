import json
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Awaitable, NamedTuple, Protocol

from loguru import logger
from mmar_mapi import Content
from pydantic import BaseModel

FileName = str
FileData = tuple[FileName, bytes]
MessageData = tuple[Content | None, FileData | None]


class ModelInfo(BaseModel):
    model: str
    caption: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    default_model: str


RequestCall = Callable[..., Awaitable[bytes | dict]]
BotConfig = NamedTuple("BotConfig", [("timeout", int)])
ResourcesConfig = NamedTuple("ResourcesConfig", [("error", str)])


def parse_headers(headers: dict | str | None) -> dict | None:
    """Parse headers from dict or JSON string.

    Args:
        headers: Headers as dict, JSON string, or None.

    Returns:
        Parsed headers dict, or None if input is falsy.
    """
    if not headers:
        return None
    if isinstance(headers, dict):
        return headers
    if isinstance(headers, str):
        try:
            return json.loads(headers)
        except Exception:
            logger.warning(f"Failed to parse headers: {headers}")
            return None
    logger.warning(f"Unexpected headers with type {type(headers)} passed: {headers}")
    return None


class MaestroClientConfigProtocol(Protocol):
    addresses__maestro: str
    res: ResourcesConfig
    headers_extra: dict[str, str] | None
    files_dir: str | None
    timeout: int
    with_retries: bool


@dataclass
class MaestroClientConfig(MaestroClientConfigProtocol):
    """Configuration for MaestroClient.

    Attributes:
        addresses__maestro: Maestro server address (default: "https://maestro.airi.net")
        res: Resources config with error message (default: ResourcesConfig(error="Server is not available"))
        headers_extra: Extra headers to include in requests (default: None)
        files_dir: Local directory for file storage, or None to use remote storage (default: None)
        timeout: Request timeout in seconds (default: 120)
        with_retries: Whether to retry failed requests (default: False)
    """

    addresses__maestro: str = "https://maestro.airi.net"
    res: ResourcesConfig = field(default_factory=lambda: ResourcesConfig(error="Server is not available"))
    headers_extra: dict[str, str] | None = None
    files_dir: str | None = None
    timeout: int = 120
    with_retries: bool = False

    @classmethod
    def create(cls, config: Any) -> "MaestroClientConfig":
        """Create a MaestroClientConfig from any config-like object.

        This handles normalization of various config types including:
        - MaestroClientConfig: Returns as-is
        - SimpleNamespace: Extracts attributes with defaults
        - Any object with matching attributes: Uses getattr with defaults

        Args:
            config: Configuration object (MaestroClientConfig, SimpleNamespace, or any object).

        Returns:
            A normalized MaestroClientConfig instance.
        """
        if isinstance(config, cls):
            return config

        addresses__maestro = getattr(config, "addresses__maestro", None) or "https://maestro.airi.net"
        headers_extra = parse_headers(getattr(config, "headers_extra", None))
        error = getattr(getattr(config, "res", SimpleNamespace()), "error", "Server is not available")
        files_dir = getattr(config, "files_dir", None)
        timeout = getattr(config, "timeout", 120)
        with_retries = getattr(config, "with_retries", False)

        return cls(
            addresses__maestro=addresses__maestro,
            res=ResourcesConfig(error=error),
            headers_extra=headers_extra,
            files_dir=files_dir,
            timeout=timeout,
            with_retries=with_retries,
        )

    @classmethod
    def from_simple_namespace(cls, ns: SimpleNamespace) -> "MaestroClientConfig":
        """Create a MaestroClientConfig from a SimpleNamespace.

        This provides backward compatibility with code that uses SimpleNamespace for configuration.

        Note: Prefer using create() which handles more cases.
        """
        return cls.create(ns)
