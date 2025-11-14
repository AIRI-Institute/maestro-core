import json
import os
from collections.abc import Iterable
from pathlib import Path


def read_json(path: Path | os.PathLike[str] | str) -> list | dict:
    return json.loads(Path(path).read_text())


def try_parse_json(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    if not text:
        return None
    if text[0] not in "{[":
        return None
    if text[-1] not in "}]":
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def try_parse_int(text: str) -> int | None:
    try:
        return int(text)
    except (ValueError, TypeError):
        return None


def try_parse_float(text: str) -> float | None:
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def try_parse_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")


def first_nonnull(obj: Iterable):
    for elem in obj:
        if elem:
            return elem
    return None


def noop(*args, **kwargs):
    pass


async def anoop(*args, **kwargs):
    pass
