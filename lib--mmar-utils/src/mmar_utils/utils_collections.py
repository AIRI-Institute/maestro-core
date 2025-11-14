from collections.abc import Callable, Iterable
from typing import Any, TypeVar

T = TypeVar("T")


def edit_object(obj: Any, editor: Callable[[T], T | None]):
    """assumed there is no null's inside obj"""
    obj_fix = editor(obj)
    if obj_fix is not None:
        return obj_fix
    if isinstance(obj, tuple):
        return tuple(edit_object(el, editor) for el in obj)
    if isinstance(obj, list):
        return list(edit_object(el, editor) for el in obj)
    return obj


def flatten(xss: Iterable[Iterable[T]]) -> list[T]:
    return [x for xs in xss for x in xs]

def take_exactly_one(elements: list[T]) -> T:
    if len(elements) != 1:
        raise ValueError(f"Exactly one element expected, found {len(elements)}: {elements}")
    return elements[0]
