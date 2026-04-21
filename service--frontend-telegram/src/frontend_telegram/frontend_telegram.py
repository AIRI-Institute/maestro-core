from datetime import timedelta, timezone
from functools import partial
from pathlib import Path

from httpx import Proxy
from loguru import logger
from mmar_mcli import MaestroClient
from telegram import BotCommand, MenuButtonCommands, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    Defaults,
    ExtBot,
    JobQueue,
    PersistenceInput,
    PicklePersistence,
)

from frontend_telegram.auth_manager import AuthManager, worker_refresh_auth_cache
from frontend_telegram.authentication import cmd_add_to_whitelist, cmd_authorize
from frontend_telegram.bots_framework import create_bot_configuration
from frontend_telegram.config import Config, ResourcesConfig, TgApplicationConfig
from frontend_telegram.custom_context import AssistantContext, BotData, ChatData, UserData
from frontend_telegram.io_async import get_or_create_loop
from frontend_telegram.io_telegram import check_telegram_available

CONTEXT_TYPES = ContextTypes(context=AssistantContext, user_data=UserData, chat_data=ChatData, bot_data=BotData)
PERSISTENCE_INPUT = PersistenceInput(bot_data=False, chat_data=False, callback_data=False, user_data=True)
AppType = Application[ExtBot[None], AssistantContext, UserData, ChatData, BotData, JobQueue]


class FrontendTelegram:
    """Main frontend Telegram bot application.

    All dependencies are injected via constructor for testability and IOC.
    """

    def __init__(
        self,
        config: Config,
        maestro_client: MaestroClient,
        auth_manager: AuthManager,
    ) -> None:
        """Initialize the FrontendTelegram application with all dependencies.

        Args:
            config: Application configuration
            maestro_client: Maestro client for backend communication
            auth_manager: Authentication manager
        """
        self._config = config
        self._maestro_client = maestro_client
        self._auth_manager = auth_manager
        self._bot_configuration = create_bot_configuration(config, maestro_client, auth_manager)

    async def _post_init(self, app: AppType) -> None:
        """Post-initialization callback for the Telegram application.

        Sets up bot commands, menu button, and descriptions.
        """
        res: ResourcesConfig = self._config.res
        commands = [
            BotCommand("/start", res.menu_start),
            BotCommand("/help", res.menu_help),
            BotCommand("/authorize", res.menu_authorize),
        ]
        await app.bot.set_my_commands(commands)

        menu_button = MenuButtonCommands()
        await app.bot.set_chat_menu_button(menu_button=menu_button)
        await app.bot.set_my_short_description(res.short_description)
        await app.bot.set_my_description(res.description)

        tg_app: TgApplicationConfig = self._config.tg_application
        logger.info(f"{tg_app.handle} started!")

    def _start_auth_workers(self) -> None:
        """Start background workers for authentication cache refresh."""
        loop = get_or_create_loop()
        loop.create_task(worker_refresh_auth_cache(self._auth_manager.users_white))

    def _create_handlers(self) -> list[object]:
        """Create all Telegram bot handlers.

        Returns:
            List of handler objects to be registered with the application.
        """
        bc = self._bot_configuration
        assert "start" in bc.commands

        entry_points = [CommandHandler(cmd, clb) for cmd, clb in bc.commands.items()]
        conv_handler = ConversationHandler(
            entry_points=entry_points,
            allow_reentry=True,  # use /start command to start a new consultation at any point
            states=bc.states,
            fallbacks=bc.fallbacks,
        )
        wrap = partial(partial, config=self._config, auth_manager=self._auth_manager)
        auth_handler = CommandHandler("authorize", wrap(cmd_authorize), has_args=1)
        white_handler = CommandHandler("white", wrap(cmd_add_to_whitelist), has_args=1)
        handlers = [
            auth_handler,
            *([white_handler] if self._config.auth.admin_path else []),
            conv_handler,
        ]
        return handlers

    def _create_application(self) -> AppType:
        """Create and configure the Telegram bot Application.

        Returns:
            Configured Telegram Application instance.
        """
        tg_app: TgApplicationConfig = self._config.tg_application
        user_persistence_path = tg_app.user_persistence_path
        Path(user_persistence_path).parent.mkdir(parents=True, exist_ok=True)
        persistence = PicklePersistence(
            filepath=user_persistence_path, store_data=PERSISTENCE_INPUT, context_types=CONTEXT_TYPES
        )
        tzinfo = timezone(timedelta(hours=tg_app.utc_delta_hours))

        builder = (
            Application.builder()
            .connect_timeout(tg_app.connect_timeout)
            .read_timeout(tg_app.read_timeout)
            .write_timeout(tg_app.write_timeout)
            .get_updates_connection_pool_size(tg_app.connection_pool_size)
            .context_types(CONTEXT_TYPES)
            .persistence(persistence)
            .token(tg_app.token)
            .defaults(Defaults(tzinfo=tzinfo, block=False))
            .post_init(self._post_init)
        )

        if tg_app.proxy_url:
            proxy = Proxy(tg_app.proxy_url)
            builder = builder.get_updates_proxy(get_updates_proxy=proxy).proxy(proxy=proxy)
            logger.info(f"Using proxy: {tg_app.proxy_url}")

        application = builder.build()
        return application

    def run(self) -> None:
        """Run the frontend Telegram bot application.

        This method:
        1. Creates the Telegram Application
        2. Registers all handlers
        3. Starts background workers (if auth is enabled)
        4. Starts polling for updates
        """
        application = self._create_application()

        handlers = self._create_handlers()
        application.add_handlers(handlers)

        if not self._config.auth.disabled:
            self._start_auth_workers()

        tg_app = self._config.tg_application
        check_telegram_available(tg_app.connect_timeout)
        application.run_polling(allowed_updates=Update.ALL_TYPES)
