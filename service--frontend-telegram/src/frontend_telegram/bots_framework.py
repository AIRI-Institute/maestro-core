from collections.abc import Awaitable, Callable
from functools import partial
from types import SimpleNamespace

from loguru import logger
from mmar_mapi import Context, make_content, make_session_id
from mmar_mapi.models.chat import _get_command, _get_text, _get_widget
from mmar_utils import remove_prefix_if_present
from pydantic import BaseModel, ConfigDict
from telegram import ReplyKeyboardRemove, Update
from telegram.error import BadRequest
from telegram.ext import BaseHandler, CallbackQueryHandler, ConversationHandler, MessageHandler

from frontend_telegram.authentication import is_authorized
from frontend_telegram.auth_manager import AuthManager
from frontend_telegram.config import Config
from frontend_telegram.custom_context import AssistantContext
from mmar_mcli import MaestroClient, MESSAGE_START, MessageData
from frontend_telegram.io_telegram import (
    extract_content,
    extract_file_data,
    select_chosen_button,
    send_action_typing,
    send_message,
    send_messages,
    send_resource,
    with_return,
)
from frontend_telegram.utils_telegram import CMD_PATTERN, make_inline_kbd, make_kbd_from_w, make_text_or_file_filter

END_DATA = "END_DATA"
WAIT_SWITCH_ENTRYPOINT = "WAIT_SWITCH_ENTRYPOINT"
NO_END_SESSION = "no_end_session"
CMD_NO_END = {"args": [NO_END_SESSION]}
CtxBot = SimpleNamespace
TelegramCallback = Callable[[Update, AssistantContext], Awaitable[int | None]]
States = dict[int | str, list[BaseHandler]]


class BotConfiguration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    states: States
    commands: dict[str, TelegramCallback]
    fallback: BaseHandler | None

    @property
    def fallbacks(self) -> list[BaseHandler]:
        return [self.fallback] if self.fallback else []


@send_action_typing
async def _process_and_response(
    upd: Update,
    context: AssistantContext,
    ctx_bot: CtxBot,
    msg_override: MessageData | None = None,
) -> None:
    if msg_override:
        msg = msg_override
    else:
        logger.debug(f"Extracting message from user_data: {context.user_data}")
        msg_content = await extract_content(upd)
        msg_file_data = await extract_file_data(upd)
        msg = msg_content, msg_file_data
    await _process_and_response_inner(upd, context, ctx_bot, msg)


async def _process_and_response_inner(
    upd: Update, context: AssistantContext, ctx_bot: CtxBot, msg: MessageData
) -> int | None:
    if ctx_bot.end_markup:
        await delete_end_markup(upd, context)

    content, _ = msg

    # when 'end session' clicked
    # todo fix: move handling clicking END_DATA to separate handler
    if (_get_command(content) or {}).get("query") == END_DATA:
        msg_data_response = make_content(text=ctx_bot.cfg.res.end_message, command=CMD_NO_END)
        await send_bot_response(upd, context, ctx_bot, msg_data_response)
        return ConversationHandler.END

    chat_context = context.user_data.chat_context
    if not chat_context.client_id:
        # todo fix: this is hotfix, should be filled before
        client_id = remove_prefix_if_present(ctx_bot.cfg.tg_application.handle, "@")
        chat_context = chat_context.model_copy(update={"client_id": client_id})

    msg_datas_response: list[MessageData] = (await ctx_bot.maestro_client.send_simple(chat_context, msg)) or []
    # only one command supported
    command: dict | None = None
    for msg_data_response in msg_datas_response:
        command = command or _get_command(msg_data_response[0])
        await send_bot_response(upd, context, ctx_bot, msg_data_response)

    command: dict = command or {}
    cmd: str | None = command.get("cmd")
    args: list = command.get("args") or []
    if not command:
        return None
    logger.info(f"Received command: {command}")

    if cmd == "poll_again":
        # this is complex and fragile logic
        # replace with regular polling in the future
        msg_data_extra = make_content(command={"cmd": "poll_again"}), None
    elif cmd == "switch":
        switch_context = args[0]
        chat_context: Context = Context.model_validate(switch_context)
        context.user_data.chat_context = chat_context
        msg_data_extra = make_content(text="/start"), None
    else:
        return

    return await _process_and_response_inner(upd, context, ctx_bot, msg=msg_data_extra)


async def send_bot_response(
    upd: Update, context: AssistantContext, ctx_bot: CtxBot, msg_data_response: MessageData
) -> None:
    content, file_data = msg_data_response
    # todo fix: maybe better text first?
    text = _get_text(content)

    await send_resource(upd, context, file_data)

    if widget := _get_widget(content):
        kbd = make_kbd_from_w(widget)
    else:
        # todo fix
        s_command = str(_get_command(content))
        if NO_END_SESSION in s_command:
            kbd = None
        else:
            kbd = ctx_bot.end_markup

    last_msg = await send_message(update=upd, context=context, text=text, kbd=kbd)
    if last_msg and kbd == ctx_bot.end_markup:
        context.user_data.prev_message_id = last_msg.id


def get_previous_markup(update: Update) -> list[list[object]] | None:
    query = update.callback_query
    if not query:
        return None
    return query.message.reply_markup.inline_keyboard


async def delete_end_markup(update: Update, context: AssistantContext) -> None:
    prev_message_id = context.user_data.get_prev_message_id()
    if not prev_message_id:
        return
    previous_markup = get_previous_markup(update)
    if previous_markup:
        return
    bot = update.get_bot()
    chat_id = update.effective_chat.id
    try:
        await bot.edit_message_reply_markup(chat_id=chat_id, message_id=prev_message_id)
    except Exception as ex:
        if isinstance(ex, BadRequest):
            logger.error(f"Failed to delete end markup: {ex}")
        else:
            logger.exception("Todo fix")


async def _answer_end_generic(upd: Update, context: AssistantContext, end_message: str) -> int:
    if query := upd.callback_query:
        await select_chosen_button(query)
    kbd = ReplyKeyboardRemove()
    await send_message(update=upd, context=context, text=end_message, kbd=kbd)
    logger.info(f"Session {context.user_data} ended")
    return ConversationHandler.END


async def _start_bot(
    upd: Update,
    context: AssistantContext,
    ctx_bot: CtxBot,
    track_id: str,
) -> None:
    if not is_authorized(upd, context, ctx_bot.auth_manager):
        return await send_messages(upd, context, ctx_bot.cfg.res.reject_access)
    chat_context = Context(
        client_id=ctx_bot.cfg.tg_application.handle[1:],
        user_id=upd.message and str(upd.message.from_user.id),
        track_id=track_id,
        session_id=make_session_id(),
    )
    logger.info(f"Created fresh chat_context: {chat_context}")
    context.user_data.chat_context = chat_context
    await _process_and_response(upd, context, ctx_bot=ctx_bot, msg_override=MESSAGE_START)


def make_fallback(end_button: str, end_message: str) -> BaseHandler | None:
    if not end_button:
        return None
    answer_end = partial(_answer_end_generic, end_message=end_message)
    fallback = CallbackQueryHandler(callback=answer_end, pattern=END_DATA)
    return fallback


def create_bot_configuration(
    config: Config,
    maestro_client: MaestroClient,
    auth_manager: AuthManager,
) -> BotConfiguration:
    commands = config.bot.commands
    end_button = config.res.end_button
    end_message = config.res.end_message

    allowed_files_extensions = config.bot.allowed_files_extensions
    f_filter = make_text_or_file_filter(exts=allowed_files_extensions)

    end_markup = make_inline_kbd([[f"{END_DATA}:{end_button}"]]) if config.bot.show_end_button else None

    ctx_bot = SimpleNamespace(
        cfg=config,
        maestro_client=maestro_client,
        auth_manager=auth_manager,
        end_markup=end_markup,
    )
    wrap = partial(partial, ctx_bot=ctx_bot)

    for cmd, track_id in commands.items():
        clb = partial(wrap(_start_bot), track_id=track_id)
        clb_fix = with_return(clb, value=cmd)
        commands[cmd] = clb_fix
    if helper := config.res.helper:
        commands["help"] = partial(send_messages, messages=[helper])

    responser = wrap(_process_and_response)
    handlers = [
        MessageHandler(f_filter, responser),
        CallbackQueryHandler(callback=responser, pattern=CMD_PATTERN),
    ]
    states = {cmd: handlers for cmd in commands.keys()}

    fallback = make_fallback(end_button, end_message)

    res = BotConfiguration(commands=commands, states=states, fallback=fallback)
    return res
