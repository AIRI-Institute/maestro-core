import json
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class Base(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    @model_validator(mode="before")
    @classmethod
    def validate_to_json(cls, value: str | Any) -> Any:
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value
