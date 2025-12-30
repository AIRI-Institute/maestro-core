import json
from pathlib import Path
from typing import Annotated, Any, Generic, ParamSpec, TypeVar

from pydantic import BaseModel, BeforeValidator, ValidationInfo

P = ParamSpec("P")
T = TypeVar("T", bound=BaseModel)


def load_file(path_field: str):
    """
    PydanticSettings validator to load a JSON file from a path specified in another field.

    It reads the file and returns a dictionary. Pydantic then validates
    this dictionary against the field's type annotation.
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


# todo eliminate
class LoadPydanticModel(Generic[T, P]):
    """
    Usage example:
    ```
    class Config(BaseSettings):
        subconfig_path: str
        subconfig: LoadPydanticModel[SubConfig, "subconfig_path"] = None
    ```
    """
    def __class_getitem__(cls, item):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("LoadPydanticModel requires exactly 2 arguments: model type and path field name")

        model_type, path_field = item
        if not isinstance(path_field, str):
            raise TypeError("Path field must be a string")

        # Create the Annotated type with our loader
        return Annotated[model_type, load_file(path_field)]
