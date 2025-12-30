from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class SetCacheI(Generic[T]):
    def refresh_if_needed(self) -> bool:
        raise NotImplementedError

    def __contains__(self, key: T) -> bool:
        raise NotImplementedError

class SetCache(SetCacheI, Generic[T]):
    def __init__(
        self,
        name: str,
        loader: Callable[[], set[T]],
        change_time_getter: Callable[[], float],
    ):
        self.name = name
        self.loader = loader
        self.change_time_getter = change_time_getter
        data_time: tuple[set[T], float] = self._refresh()
        self.data, self.last_change_time = data_time

    def _refresh(self) -> tuple[set[T], float]:
        data, last_change_time = self.loader(), self.change_time_getter()
        if not isinstance(data, set):
            raise ValueError(f"Bad data: expected set, found {type(data)}: {data}")
        if not isinstance(last_change_time, float):
            raise ValueError(f"Bad last_change_time: {last_change_time} :: {type(last_change_time)}")
        return data, last_change_time

    def refresh_if_needed(self) -> bool:
        if self.last_change_time < self.change_time_getter():
            self.data, self.last_change_time = self._refresh()
            return True
        return False

    def __contains__(self, key: T) -> bool:
        res = key in self.data
        return res


class SetCacheFixed(SetCacheI[T]):
    def __init__(self, fixed: bool):
        self.fixed = fixed

    def refresh_if_needed(self) -> bool:
        return False

    def __contains__(self, key: T) -> bool:
        return self.fixed

SET_CACHE_ALWAYS_YES = SetCacheFixed(True)
SET_CACHE_ALWAYS_NO = SetCacheFixed(False)
