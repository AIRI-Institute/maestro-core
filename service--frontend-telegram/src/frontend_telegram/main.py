from datetime import timedelta, timezone
from functools import partial
from pathlib import Path
from warnings import filterwarnings

from loguru import logger
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
from telegram.warnings import PTBUserWarning

from frontend_telegram.auth_manager import worker_fill_user_id, worker_refresh_auth_cache
from frontend_telegram.authentication import cmd_add_to_whitelist, cmd_authorize
from frontend_telegram.bots_framework import BotConfiguration, create_bot_configuration
from frontend_telegram.config import Config
from frontend_telegram.custom_context import AssistantContext, BotData, ChatData, UserData
from frontend_telegram.dependencies import Dependencies
from frontend_telegram.io_async import get_or_create_loop
from frontend_telegram.io_telegram import check_telegram_available

# remove per_message=False CallbackQueryHandler warning
filterwarnings(action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning)

CONTEXT_TYPES = ContextTypes(context=AssistantContext, user_data=UserData, chat_data=ChatData, bot_data=BotData)
PERSISTENCE_INPUT = PersistenceInput(bot_data=False, chat_data=False, callback_data=False, user_data=True)
AppType = Application[ExtBot[None], AssistantContext, UserData, ChatData, BotData, JobQueue]


async def post_init(app: AppType, config: Config) -> None:
    commands = [
        BotCommand("/start", config.res.menu_start),
        BotCommand("/help", config.res.menu_help),
        BotCommand("/authorize", config.res.menu_authorize),
    ]
    await app.bot.set_my_commands(commands)

    menu_button = MenuButtonCommands()
    await app.bot.set_chat_menu_button(menu_button=menu_button)
    await app.bot.set_my_short_description(config.res.short_description)
    await app.bot.set_my_description(config.res.description)

    logger.info(f"{config.tg_application.handle} started!")


def start_auth_workers(deps: Dependencies) -> None:
    loop = get_or_create_loop()
    loop.create_task(worker_refresh_auth_cache(deps.auth_manager.users_white))
    telethon_client = deps.telethon_client
    if not telethon_client:
        return
    user_id_loader = telethon_client.get_user_id_by_username
    whitelist_path = Path(deps.config.auth.whitelist_path)
    loop.create_task(worker_fill_user_id(user_id_loader, whitelist_path))


def create_handlers(deps: Dependencies, bc: BotConfiguration) -> list[object]:
    config = deps.config
    assert "start" in bc.commands
    entry_points = [CommandHandler(cmd, clb) for cmd, clb in bc.commands.items()]
    conv_handler = ConversationHandler(
        entry_points=entry_points,
        allow_reentry=True,  # use /start command to start a new consultation at any point
        states=bc.states,
        fallbacks=bc.fallbacks,
    )
    wrap = partial(partial, config=config, auth_manager=deps.auth_manager)
    auth_handler = CommandHandler("authorize", wrap(cmd_authorize), has_args=1)
    white_handler = CommandHandler("white", wrap(cmd_add_to_whitelist), has_args=1)
    handlers = [
        auth_handler,
        *([white_handler] if config.auth.admin_path else []),
        conv_handler,
    ]
    return handlers


def serve(config: Config | None = None) -> None:
    deps = Dependencies().override_config(config)
    config = deps.config
    bc: BotConfiguration = create_bot_configuration(deps)

    user_persistence_path = config.tg_application.user_persistence_path
    Path(user_persistence_path).parent.mkdir(parents=True, exist_ok=True)
    persistence = PicklePersistence(
        filepath=user_persistence_path, store_data=PERSISTENCE_INPUT, context_types=CONTEXT_TYPES
    )
    tzinfo = timezone(timedelta(hours=config.tg_application.utc_delta_hours))
    application = (
        Application.builder()
        .connect_timeout(config.tg_application.connect_timeout)
        .read_timeout(config.tg_application.read_timeout)
        .write_timeout(config.tg_application.write_timeout)
        .get_updates_connection_pool_size(config.tg_application.connection_pool_size)
        .context_types(CONTEXT_TYPES)
        .persistence(persistence)
        .token(config.tg_application.token)
        .defaults(Defaults(tzinfo=tzinfo, block=False))
        .post_init(partial(post_init, config=config))
        .build()
    )

    handlers = create_handlers(deps, bc)
    application.add_handlers(handlers)

    if not config.auth.disabled:
        start_auth_workers(deps)

    check_telegram_available(config.tg_application.connect_timeout)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    serve()
