"""
Generic Pydantic Field Loader

Load nested Pydantic models from JSON/TOML files or environment variables.

Usage:
    ```python
    class Config(BaseSettings):
        config_path: str | None = None
        config: LoadPydanticModel[SubConfig, "config_path"] = Field(default_factory=SubConfig)
    ```

Features:
- Loads from JSON/TOML file when path is provided
- Falls back to environment variable loading when path is None
- Auto-detects file format by extension
"""

import json
from pathlib import Path
from typing import Annotated, Any, Generic, TypeVar

import tomllib

from pydantic import BaseModel, BeforeValidator, ValidationInfo

T = TypeVar("T", bound=BaseModel)

# Supported file formats
_FORMATS = {".json": json.loads, ".toml": tomllib.loads}


def _validate_type_args(item: tuple[Any, ...]) -> tuple[type[T], str]:
    """Validate and extract type arguments for LoadPydanticModel."""
    if not isinstance(item, tuple) or len(item) != 2:
        raise TypeError(
            f"LoadPydanticModel requires [ModelType, 'path_field'], got {item!r}"
        )

    model_type, path_field = item

    if not isinstance(path_field, str):
        raise TypeError(
            f"Path field must be a string, got {type(path_field).__name__}"
        )

    # Validate model_type is a BaseModel (skip for ForwardRef)
    try:
        if not (isinstance(model_type, type) and issubclass(model_type, BaseModel)):
            raise TypeError(
                f"Model type must be a BaseModel subclass, got {model_type!r}"
            )
    except TypeError:
        pass  # ForwardRef or other special type

    return model_type, path_field


def _load_file(path_value: str) -> dict[str, Any]:
    """Load and parse a JSON or TOML file."""
    path = Path(path_value)

    if not path.exists():
        raise ValueError(f"File not found: {path_value}")

    suffix = path.suffix.lower()

    if suffix not in _FORMATS:
        raise ValueError(
            f"Unsupported format: {suffix}. Supported: {', '.join(_FORMATS)}"
        )

    try:
        content = path.read_text()
        return _FORMATS[suffix](content)
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as e:
        kind = "JSON" if suffix == ".json" else "TOML"
        raise ValueError(f"Invalid {kind} in {path_value}") from e


def _make_validator(path_field: str) -> Any:
    """Create a BeforeValidator for loading from the specified path field."""
    def validator(v: Any, info: ValidationInfo) -> Any:
        path_value = info.data.get(path_field)

        if not path_value:
            return v  # Allow default/env var loading

        try:
            return _load_file(str(path_value))
        except ValueError as e:
            raise ValueError(
                f"Failed to load {info.field_name} from {path_field}={path_value}: {e}"
            ) from e

    return BeforeValidator(validator)


def load_file(path_field: str, *, optional: bool = False) -> Any:
    """
    Create a validator to load a JSON/TOML file from a path field.

    Args:
        path_field: Name of the field containing the file path
        optional: If True, returns original value when path_field is None/empty

    Returns:
        A BeforeValidator for use with Annotated
    """
    return _make_validator(path_field)


class LoadPydanticModel(Generic[T]):
    """
    Generic loader for Pydantic models from file path or environment variables.

    Usage:
        ```python
        config: LoadPydanticModel[ConfigModel, "config_path"]
        ```
    """

    def __class_getitem__(cls, item: tuple[type[T], str]) -> type:
        model_type, path_field = _validate_type_args(item)
        return Annotated[model_type, _make_validator(path_field)]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "LoadPydanticModel cannot be instantiated. "
            "Use as a type annotation: LoadPydanticModel[ModelType, 'path_field']"
        )


