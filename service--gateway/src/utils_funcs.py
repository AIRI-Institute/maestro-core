from collections.abc import Callable
from functools import partial, update_wrapper
from inspect import Signature, signature
from typing import Any, TypeVar

T = TypeVar("T")


def partial_hide(func: Callable[..., T], /, **fixed_kwargs: Any) -> Callable[..., T]:
    """
    Like functools.partial, but removes frozen arguments
    from the function signature so FastAPI doesn't see them.
    """
    # create a partial first
    f: Callable = partial(func, **fixed_kwargs)
    update_wrapper(f, func)

    # build new signature
    sig = signature(func)
    params = [p for name, p in sig.parameters.items() if name not in fixed_kwargs]
    f.__signature__ = Signature(params)  # type: ignore[attr-defined]

    return f
