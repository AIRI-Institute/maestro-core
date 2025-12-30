import os
from functools import cached_property

from loguru import logger
from mmar_mcli import MaestroClient, MaestroClientDummy
from mmar_ptag import init_logger

from frontend_telegram.auth_manager import AuthManager
from frontend_telegram.config import Config, load_config
from frontend_telegram.telethon_client import TelethonClient


class Dependencies:
    def __init__(self) -> None:
        self._config_override: Config | None = None

    def override_config(self, config: Config | None) -> "Deps":
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
    def maestro_client(self) -> MaestroClient:
        if self.config.bot.is_dummy_maestro:
            logger.error("Initializing dummy maestro")
            return MaestroClientDummy(self.config)
        else:
            return MaestroClient(self.config)

    @cached_property
    def telethon_client(self) -> TelethonClient | None:
        if not self.config.telethon.api_bot_token:
            return None
        return TelethonClient(self.config.telethon)

    @cached_property
    def auth_manager(self) -> AuthManager:
        return AuthManager(self.config)
