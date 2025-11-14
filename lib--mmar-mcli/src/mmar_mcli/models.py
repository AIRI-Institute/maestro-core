from collections.abc import Callable
from typing import Awaitable, NamedTuple, Protocol

from mmar_mapi import Content

FileName = str
FileData = tuple[FileName, bytes]
MessageData = tuple[Content | None, FileData | None]

RequestCall = Callable[..., Awaitable[bytes | dict]]
BotConfig = NamedTuple("BotConfig", [("timeout", int)])
ResourcesConfig = NamedTuple("ResourcesConfig", [("error", str)])


class MaestroConfig(Protocol):
    addresses__maestro: str
    res: ResourcesConfig
    headers_extra: dict[str, str] | None
    files_dir: str | None
    timeout: int
