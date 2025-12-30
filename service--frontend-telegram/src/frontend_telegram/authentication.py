from pathlib import Path

from loguru import logger
from telegram import Update

from frontend_telegram.auth_manager import add_to_authlist
from frontend_telegram.auth_manager import AuthManager
from frontend_telegram.config import Config
from frontend_telegram.custom_context import AssistantContext
from frontend_telegram.utils_telegram import get_user_id_username


def is_authorized(update: Update, context: AssistantContext, auth_manager: AuthManager) -> bool:
    if context.user_data.is_authorized:
        return True
    user_id_username = get_user_id_username(update)
    if auth_manager.is_authorized(user_id_username):
        context.user_data.is_authorized = True
        return True
    return False


async def cmd_authorize(update: Update, context: AssistantContext, config: Config, auth_manager: AuthManager) -> None:
    logger.info(f"Attempt to authorize: {context.user_data}")
    if context.user_data.is_authorized:
        await update.effective_user.send_message(config.res.auth_repeat)
        return

    input_password = context.args[0]
    if input_password != config.auth.tg_password:
        await update.effective_user.send_message(config.res.auth_failure)
        return

    context.user_data.is_authorized = True
    await update.effective_user.send_message(config.res.auth_success)


async def cmd_add_to_whitelist(
    update: Update, context: AssistantContext, config: Config, auth_manager: AuthManager
) -> None:
    user_id_username = get_user_id_username(update)
    if not auth_manager.is_admin(user_id_username):
        return
    whitelist_path = Path(config.auth.whitelist_path)
    username = context.args[0] or ""
    logger.info(f"Attempt to add username to whitelist: `{username}`")
    msg = add_to_authlist(whitelist_path, username)
    logger.info(f"Result of #add_to_authlist: {msg}")
    await update.effective_user.send_message(msg)
