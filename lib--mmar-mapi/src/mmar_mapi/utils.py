from itertools import islice
from collections.abc import Iterable
from datetime import datetime


def make_session_id(with_millis=False) -> str:
    dt = datetime.now()
    fmt = "%Y-%m-%d--%H-%M-%S-%f" if with_millis else "%Y-%m-%d--%H-%M-%S"
    return dt.strftime(fmt)


def chunked(items: Iterable, n) -> list:
    "behavior like in more_itertools.chunked"
    iterator = iter(items)
    res = []
    while chunk := list(islice(iterator, n)):
        res.append(chunk)
    return res
