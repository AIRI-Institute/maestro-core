from dishka import Provider, Scope, provide
from loguru import logger
from mmar_mapi import Context
from mmar_mcli import MaestroClient, MaestroClientDummy, MessageData
from mmar_mimpl import init_logger

from frontend_telegram.auth_manager import AuthManager
from frontend_telegram.config import AuthConfig, Config
from frontend_telegram.frontend_telegram import FrontendTelegram


class IOC(Provider):
    """Provider for shared/external dependencies."""

    scope = Scope.APP

    @provide
    def maestro_client(self, config: Config) -> MaestroClient:
        """Create MaestroClient."""
        if config.bot.is_dummy_maestro:
            logger.error("Initializing dummy maestro")
            mc = MaestroClientDummy(config)

            # todo move this logic to the mmar_mcli: support passing callback `msg_data -> [msg_data]`
            msg = config.bot.dummy_message
            if msg:

                async def send_simple(ctx: Context, msg_data: MessageData) -> list[MessageData]:
                    return [(msg, None)]

                mc.send_simple = send_simple
        else:
            mc = MaestroClient(config)
        return mc


class IOCLocal(Provider):
    """Provider for local application-specific dependencies."""

    scope = Scope.APP

    @provide
    def config(self) -> Config:
        """Load and return configuration."""
        config = Config.load()
        init_logger(config.logger.level)
        logger.debug(f"Config: {config}")
        return config

    @provide
    def auth_config(self, config: Config) -> AuthConfig:
        """Provide auth configuration."""
        return config.auth

    @provide
    def auth_manager(self, auth_config: AuthConfig) -> AuthManager:
        """Create AuthManager."""
        return AuthManager(auth_config)

    @provide
    def frontend_telegram(
        self,
        config: Config,
        maestro_client: MaestroClient,
        auth_manager: AuthManager,
    ) -> FrontendTelegram:
        """Create FrontendTelegram with all its dependencies."""
        return FrontendTelegram(
            config=config,
            maestro_client=maestro_client,
            auth_manager=auth_manager,
        )
