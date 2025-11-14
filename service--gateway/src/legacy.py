import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from mmar_mapi import AIMessage, Context, FileStorage, HumanMessage, ResourceId
from pydantic import BaseModel

from src.models import FileData, NamedResourceId


# todo eliminate: move to FileStorage
def upload_resource_maybe(file_storage: FileStorage, f_data: FileData | None) -> NamedResourceId | None:
    if not f_data:
        return None
    fname, content = f_data
    resource_id: ResourceId = file_storage.upload(content, fname=fname)
    # todo fix: why? we don't sure that final resource name is `fname`?
    resource_name = file_storage.get_fname(resource_id) or fname
    return resource_name, resource_id


# todo eliminate: move to FileStorage
def as_file_data(file_storage: FileStorage, resource_id: ResourceId) -> FileData:
    res_name = file_storage.get_fname(resource_id)
    if not res_name:
        res_ext = resource_id.rsplit(".", 1)[-1]
        res_name = f"result.{res_ext}"
    res_bytes = file_storage.download(resource_id)
    return (res_name, res_bytes)


T = TypeVar("T")


def make_async(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """
    Decorator to convert a blocking function into an async one by running it in a thread.

    Usage:
        @make_async
        def blocking_func() -> Response:
            ...

        async def call_it():
            result = await blocking_func()  # Now it's async
    """

    async def async_wrapper(*args: Any, **kwargs: Any) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)

    return async_wrapper


class ChatRequestOld(BaseModel):
    context: Context
    messages: list[HumanMessage]


class ChatResponseOld(ChatRequestOld):
    response_messages: list[AIMessage]
