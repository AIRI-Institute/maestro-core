"""Configuration loader for the unified LLM proxy service."""

from typing import Annotated, Any, Literal, Union

from mmar_mimpl import LoadPydanticModel, SettingsModel
from pydantic import BaseModel, Field


class BaseConnectionConfig(BaseModel):
    """Base configuration for all connection types."""

    max_concurrent: int | None = None


class OpenAIConnectionConfig(BaseConnectionConfig):
    """Configuration for OpenAI-compatible connections."""

    api_type: Literal["openai"] = "openai"
    api_base: str
    api_key: str


class GigachatConnectionConfig(BaseConnectionConfig):
    """Configuration for GigaChat connections."""

    api_type: Literal["gigachat"] = "gigachat"
    api_base: str | None = None
    api_key: str | None = None
    verify_ssl: bool = True

    scope: str | None = None
    auth_url: str | None = None
    access_token: str | None = None
    authorization: str | None = None  # OAuth authorization code
    user: str | None = None
    password: str | None = None
    profanity_check: bool | None = None
    # Embeddings: 1024 dimensions, EmbeddingsGigaR: 2560 dimensions
    embeddings_model: Literal["Embeddings", "EmbeddingsGigaR"] = "Embeddings"


class AnthropicConnectionConfig(BaseConnectionConfig):
    """Configuration for Anthropic API connections."""

    api_type: Literal["anthropic"] = "anthropic"
    api_base: str
    api_key: str


# Discriminated union for connection configs
ConnectionConfig = Annotated[
    Union[
        OpenAIConnectionConfig,
        GigachatConnectionConfig,
        AnthropicConnectionConfig,
    ],
    Field(discriminator="api_type"),
]


class ModelInfo(BaseModel):
    """Metadata and configuration for a model."""

    caption: str | None = None
    owned_by: str | None = None
    max_concurrent: int | None = None
    default: bool | None = None
    default_embeddings: bool | None = None


class LLMConfig(BaseModel):
    connections: dict[str, ConnectionConfig] = Field(default_factory=dict)
    routing: dict[str, str] = Field(default_factory=dict)
    model_info: dict[str, ModelInfo] = Field(default_factory=dict)
    groups: dict[str, Any] = Field(default_factory=dict)
    middleware: list[dict[str, Any]] = Field(default_factory=list)
    fallback_to_default: bool = True


class Config(SettingsModel):
    llm_config_path: str | None = None
    llm_config: LoadPydanticModel[LLMConfig, "llm_config_path"] = Field(default_factory=LLMConfig)
