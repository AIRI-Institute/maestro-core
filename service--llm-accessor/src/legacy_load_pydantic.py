# todo move to mmar_utils or mmar_mapi
import json
from pathlib import Path
from typing import Annotated, Any, Generic, TypeVar

from pydantic import AfterValidator, BaseModel, BeforeValidator, ValidationInfo
from pydantic_settings import BaseSettings
from typing_extensions import ParamSpec  # or typing.ParamSpec (Python 3.10+)

# === 1. Your Data Models (Unchanged) ===
# ... (EntrypointConfig, Entrypoints, EntrypointsConfig) ...

# === 2. The Custom Generic Type ===
P = ParamSpec("P")
T = TypeVar("T", bound=BaseModel)


def load_file(path_field: str):
    """
    Pydantic validator to load a JSON file from a path specified in another field.

    It reads the file and returns a dictionary. Pydantic then validates
    this dictionary against the field's type annotation (e.g., EntrypointsConfig).
    """

    def validator(v: Any, info: ValidationInfo) -> dict:
        path_value = info.data.get(path_field)
        if path_value is None:
            raise ValueError(f"Field '{path_field}' required to load {info.field_name} but not found")
        try:
            content = Path(path_value).read_text()
            return json.loads(content)
        except FileNotFoundError as e:
            raise ValueError(f"File not found at path: {path_value}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file: {path_value}") from e

    return BeforeValidator(validator)


class LoadPydantic(Generic[T, P]):
    def __class_getitem__(cls, item):
        # Parse the generic arguments
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("LoadPydantic requires exactly 2 arguments: model type and path field name")

        model_type, path_field = item
        if not isinstance(path_field, str):
            raise TypeError("Path field must be a string")

        # Create the Annotated type with our loader
        return Annotated[model_type, load_file(path_field)]
