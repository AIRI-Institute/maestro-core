from typing import Tuple, TypeVar


Error = TypeVar("Error")
Result = TypeVar("Result")
Either = Tuple[Error, None] | Tuple[None, Result]
