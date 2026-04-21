import os
from typing import Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = "ENV_FILE"


def _format_env_value(value: Any) -> str:
    """Format a value for env file output."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return str(value)
    return str(value)


def _model_dump_env(model: BaseModel, delimiter: str = "__") -> str:
    """
    Dump a Pydantic model as environment variables in .env format.

    Args:
        model: The Pydantic model to dump
        delimiter: Delimiter for nested keys (default: "__")

    Returns:
        String in .env file format (KEY=VALUE)
    """
    lines: list[str] = []

    def flatten(obj: Any, parts: list[str]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten(v, parts + [str(k).upper()])
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                flatten(v, parts + [str(i)])
        elif obj is None:
            return
        else:
            key = delimiter.join(parts)
            lines.append(f"{key}={_format_env_value(obj)}")

    flatten(model.model_dump(), [])

    return "\n".join(lines)


class SettingsModel(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")
    version: str = "dev"

    @classmethod
    def load(cls, env_file=None) -> "SettingsModel":
        env_file = env_file or os.getenv(ENV_FILE)
        return cls(_env_file=env_file)  # type: ignore[call-arg]

    def model_dump_env(self, delimiter: str = "__") -> str:
        """
        Dump all settings as environment variables in .env format.

        Args:
            delimiter: Delimiter for nested keys (default: "__")
        """
        return _model_dump_env(self, delimiter)
