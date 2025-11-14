from functools import cached_property

from loguru import logger
from mmar_mapi import FileStorage
from mmar_mapi.api import ChatManagerAPI
from mmar_ptag import init_logger, ptag_client

from src.chat_storage import ChatStorage
from src.config import Config, load_config
from src.gateway import Gateway


class Deps:
    def __init__(self) -> None:
        self._config_override: Config | None = None

    def override_config(self, config: Config) -> "Deps":
        self._config_override = config
        return self

    @cached_property
    def config(self) -> Config:
        if self._config_override:
            return self._config_override
        config: Config = load_config()
        init_logger(config.logger.level)
        logger.debug(f"Config: {config}")
        return config

    @cached_property
    def chat_storage(self) -> ChatStorage:
        return ChatStorage(
            logs_dir=self.config.db_path,
            logs_dir_archived=self.config.archive_path,
        )

    @cached_property
    def file_storage(self) -> FileStorage:
        return FileStorage(self.config.files_dir)

    @cached_property
    def chat_manager(self) -> ChatManagerAPI:
        addresses__chat_manager = self.config.addresses.chat_manager
        return ptag_client(ChatManagerAPI, addresses__chat_manager)

    @cached_property
    def gateway(self) -> Gateway:
        return Gateway(self.config, self.file_storage, self.chat_storage, self.chat_manager)
