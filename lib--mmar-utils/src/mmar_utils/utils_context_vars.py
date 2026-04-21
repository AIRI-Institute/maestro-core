from collections.abc import Callable
from contextvars import ContextVar
from typing import TypeVar

C = TypeVar("C")
ContextGetSet = tuple[Callable[[], C], Callable[[C], None]]


def get_getter_and_setter(var: ContextVar[C]) -> ContextGetSet:
    getter = var.get

    def setter(value: C | None) -> None:
        if value is not None:
            var.set(value)

    return getter, setter
