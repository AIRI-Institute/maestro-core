from collections.abc import AsyncIterator
from urllib.parse import unquote, urlsplit

from dishka import Provider, Scope, from_context, provide
from mmar_mapi import FileStorage
from mmar_mapi.services import ChatManagerAPI
from mmar_mcli import MaestroClientI
from mmar_ptag import ptag_client
from openai import OpenAI

from gateway.chat_storage import ChatStorageAPI, ChatStorageFS
from gateway.chat_storage_sql import ChatStorageSQL
from gateway.config import Config
from gateway.db import SqlAlchemyDatabase
from gateway.maestro_gateway import MaestroGateway


class IOC(Provider):
    """IoC dependencies with APP scope."""

    scope = Scope.APP

    @provide
    def config(self) -> Config:
        return from_context(provides=Config, scope=Scope.APP)  # type: ignore[return-value]

    @provide
    async def chat_storage(self, config: Config) -> AsyncIterator[ChatStorageAPI]:
        storage_config = config.chat_storage
        parsed_uri = urlsplit(storage_config.uri)
        scheme = parsed_uri.scheme
        raw_extra = storage_config.extra or {}

        if scheme in ("", "file"):
            unknown_keys = set(raw_extra) - {"archive_dir"}
            if unknown_keys:
                raise ValueError("Invalid chat_storage.extra for fs storage")

            archive_dir = raw_extra.get("archive_dir")
            if archive_dir is not None and not isinstance(archive_dir, str):
                raise ValueError("Invalid chat_storage.extra for fs storage")

            if scheme == "":
                uri_path = storage_config.uri
            else:
                if parsed_uri.query or parsed_uri.fragment:
                    raise ValueError("chat_storage file URI must not contain query or fragment")
                if parsed_uri.netloc not in ("", "localhost"):
                    raise ValueError("chat_storage file URI must not specify a remote host")
                if not parsed_uri.path:
                    raise ValueError("chat_storage file URI must include a path")
                uri_path = unquote(parsed_uri.path)

            yield ChatStorageFS(
                logs_dir=uri_path,
                logs_dir_archived=archive_dir,
            )
            return

        if scheme not in {"postgresql", "postgresql+asyncpg"}:
            raise ValueError(f"Unsupported chat_storage URI scheme: {scheme!r}")
        if raw_extra:
            raise ValueError("Invalid chat_storage.extra for postgresql storage")

        db = SqlAlchemyDatabase(url=storage_config.uri)
        try:
            yield ChatStorageSQL(db=db)
        finally:
            await db.dispose()

    @provide
    def file_storage(self, config: Config) -> FileStorage:
        return FileStorage(config.files_dir)

    @provide
    def chat_manager(self, config: Config) -> ChatManagerAPI:
        addresses__chat_manager = config.addresses.chat_manager
        return ptag_client(ChatManagerAPI, addresses__chat_manager)

    @provide
    def oclient(self, config: Config) -> OpenAI:
        return OpenAI(base_url=config.openai_api_base, api_key=config.openai_api_key)

    maestro_gateway: MaestroGateway = provide(MaestroGateway)  # type: ignore[assignment]
    maestro_client: MaestroClientI = provide(MaestroGateway, provides=MaestroClientI)
