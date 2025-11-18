from pydantic import BaseModel, AfterValidator
from typing import Annotated


def check_corresponding_keys(dictionary):
    for key, v in dictionary.items():
        if key == v.key:
            continue
        raise ValueError(f"mismatch of {key=} and {v.key=}")
    return dictionary


class EntrypointConfig(BaseModel):
    key: str
    name: str
    caption: str
    args: dict[str, str | bool | int]


Entrypoints = Annotated[dict[str, EntrypointConfig], AfterValidator(check_corresponding_keys)]


class EntrypointsConfig(BaseModel):
    entrypoints: Entrypoints
    default_entrypoint_key: str | None = None
    default_image_entrypoint_key: str | None = None
    default_file_entrypoint_key: str | None = None
    warmup: bool = False


ENTRYPOINTS_CONFIG_EMPTY = EntrypointsConfig(entrypoints={})
