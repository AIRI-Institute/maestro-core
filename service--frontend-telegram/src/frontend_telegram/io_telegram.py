import socket
import time
from collections.abc import Callable
from functools import partial, wraps
from io import BytesIO
from typing import Awaitable, TypeVar

from loguru import logger
from mmar_mapi import Content, make_content
from mmar_mcli import FileData
from mmar_utils import chunk_respect_semantic, edit_object, on_error_log_and_none, retry_on_ex
from telegram import InlineKeyboardMarkup, InputFile, Message, ReplyKeyboardRemove, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, NetworkError, TimedOut

from frontend_telegram.custom_context import AssistantContext
from frontend_telegram.utils_telegram import edit_button

MAX_MESSAGE_SIZE = 4096
ATTEMPTS = 5
WAIT_SECONDS = 3
# BadRequest is derived from NetworkError and usually happen due to bugs
# so we don't want to catch it
TG_CATCH = (TimedOut, NetworkError)
TG_NOCATCH = BadRequest
CHECKED = "✅"
T = TypeVar("T")
MESSAGE_IS_NOT_MODIFIED_ERR = "Message is not modified: specified new message content and reply markup are exactly the same as a current content and reply markup of the message"


async def extract_content(update: Update) -> Content | None:
    query = update.callback_query
    if query:
        data = query.data
        await select_chosen_button(query)
        await query.answer()
        logger.info(f"Catched user callback: {data}")
        return make_content(command={"query": data})

    text = update.message and (update.message.text or update.message.caption)
    # `or None` to prevent empty content
    return make_content(text=text) or None


async def extract_file_data(update: Update) -> FileData | None:
    if update.message is None:
        return None
    file_data = await try_extract_file_bytes(update.message)
    image = await try_extract_image_bytes(update.message)
    if file_data and image:
        logger.error("Assumption that `file_data` and `image` can not goes together broken!")
    return file_data or image


def _send_action(action: str) -> Callable:
    """Sends `action` while processing func command."""

    def decorator(func: Callable[[...], T]) -> Callable:
        @wraps(func)
        async def command_func(update: Update, context: AssistantContext, *args: object, **kwargs: object) -> T:
            chat_id = update.effective_message.chat_id
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=action)
            except TG_NOCATCH:
                logger.warning("Telegram error, skipping `send_chat_action`...")
            return await func(update, context, *args, **kwargs)

        return command_func

    return decorator


send_action_typing = _send_action(ChatAction.TYPING)


def _check_host_with_latency(hostname: str, port: int = 80, timeout: int = 5) -> tuple[bool, float | None]:
    start_time = time.time()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((hostname, port))
        s.close()
        end_time = time.time()
        latency = end_time - start_time
        return True, latency
    except Exception as e:
        print(f"Error connecting to {hostname}: {e}")
        return False, None


def check_telegram_available(connect_timeout: int) -> None:
    hostname = "api.telegram.org"
    reachable, latency = _check_host_with_latency(hostname, timeout=connect_timeout)
    if reachable:
        logger.info(f"{hostname} is reachable. Latency: {latency:.2f} seconds")
    else:
        logger.error(f"{hostname} is not reachable.")


@on_error_log_and_none(logger.exception, "TODO FIX, this error should not happen")
async def try_extract_image_bytes(message: Message) -> FileData | None:
    photo = message.photo
    if not photo:
        return None
    image = await photo[-1].get_file()
    image = await image.download_as_bytearray()
    image_bytes = BytesIO(image).getvalue()
    logger.info("Image loaded")
    return "tg_jpeg.jpeg", image_bytes


@on_error_log_and_none(logger.exception, "TODO FIX, this error should not happen")
async def try_extract_file_bytes(message: Message) -> FileData | None:
    if message.document is None:
        return None
    media_group_id = message.media_group_id
    if media_group_id:
        # telegram-bot-framework + Telegram API restrictions.. :(
        # https://github.com/python-telegram-bot/python-telegram-bot/wiki/Frequently-requested-design-patterns#how-do-i-deal-with-a-media-group
        # -- the only one link which doesn't help
        # I tries approach with downloading history via telethon, but it's impossible:
        # -- telethon.errors.rpcerrorlist.BotMethodInvalidError: The API access for bot users is restricted. The method you tried to invoke cannot be executed as a bot (caused by GetHistoryRequest)
        logger.warning(
            f"Detected media_group_id: {media_group_id}, probably user sent many documents, but we can parse only first"
        )
    doc_file_name = message.document.file_name
    doc_file = await message.document.get_file()
    file_bytearray = await doc_file.download_as_bytearray()
    file_bytes = BytesIO(file_bytearray).getvalue()
    logger.info(f"Received document: {doc_file_name}, size: {len(file_bytes)} bytes")
    return doc_file_name, file_bytes


async def send_resource(update: Update, context: AssistantContext, file_data: FileData | None) -> Message | None:
    if file_data is None:
        return
    resource_fname, resource_bytes = file_data
    resource_ext = resource_fname.split(".", 1)[-1]
    sz = len(resource_bytes)
    logger.info(f"Going to send resource '{resource_fname}', size: {sz}")
    if resource_ext in {"jpg", "jpeg"}:
        file_document = InputFile(obj=resource_bytes)
        message = await update.effective_user.send_photo(photo=file_document)
    else:
        resource_bytes_io = BytesIO(resource_bytes)
        resource_bytes_io.name = resource_fname
        file_document = InputFile(resource_bytes_io)
        message = await update.effective_user.send_document(document=file_document)
    return message


async def send_messages(
    update: Update,
    context: AssistantContext,
    messages: list[str],
    kbd: InlineKeyboardMarkup | None = None,
) -> None:
    ii_last = len(messages) - 1
    for ii, msg in enumerate(messages):
        if not msg:
            continue
        reply_markup = kbd if ii == ii_last else None
        await send_message(
            update=update,
            context=context,
            text=msg,
            kbd=reply_markup,
        )


def detect_parse_mode(text: str) -> ParseMode | None:
    if "<b>" in text:
        return ParseMode.HTML
    if "**" in text or "`" in text:
        return ParseMode.MARKDOWN
    return None


@retry_on_ex(attempts=ATTEMPTS, wait_seconds=WAIT_SECONDS, catch=TG_CATCH, nocatch=TG_NOCATCH, logger=logger.warning)
async def send_message(
    update: Update,
    context: AssistantContext,
    text: str,
    kbd: InlineKeyboardMarkup | None = None,
) -> Message | None:
    if not text:
        return None

    kbd = kbd or ReplyKeyboardRemove()
    parse_mode = detect_parse_mode(text)
    chunks = chunk_respect_semantic(text, MAX_MESSAGE_SIZE)
    ii_last = len(chunks) - 1
    for ii, chunk in enumerate(chunks):
        reply_markup = None if ii < ii_last else kbd
        send_kwargs = dict(text=chunk, reply_markup=reply_markup, parse_mode=parse_mode)
        try:
            msg = await update.effective_user.send_message(**send_kwargs)
        except Exception:
            if parse_mode is None:
                raise
            # fallback to sending message with parse_mode = None
            # introducing normal `detect_parse_mode` is expensive, so it's cheaper ( in average ) to assume that text is valid markdown
            send_kwargs["parse_mode"] = None
            msg = await update.effective_user.send_message(**send_kwargs)
    # todo fix: need really normal fallback: maybe just send "Что-то пошло не так..."
    return msg


async def select_chosen_button(query: object) -> None:
    await query.answer()
    selected = query.data
    inline_keyboard: InlineKeyboardMarkup = query.message.reply_markup.inline_keyboard
    add_mark = partial("{0} {1}".format, CHECKED)
    button_editor = partial(edit_button, selected=selected, text_callback=add_mark)
    inline_keyboard_fix = edit_object(inline_keyboard, editor=button_editor)
    logger.debug(f"Updating keyboard: {inline_keyboard} -> {inline_keyboard_fix}")
    kbd = InlineKeyboardMarkup(inline_keyboard_fix)
    try:
        await query.edit_message_reply_markup(reply_markup=kbd)
    except BadRequest as ex:
        if str(ex) == MESSAGE_IS_NOT_MODIFIED_ERR:
            # because can't update keyboard twice...
            logger.warning("Fix this exeption later, now skipping...")
            return
        logger.exception(f"Failed to select button with selected='{selected}'")


def with_return(func: Callable[[Update, AssistantContext], Awaitable], value: T) -> Callable:
    @wraps(func)
    async def decorator(update: Update, context: AssistantContext) -> T:
        await func(update, context)
        return value

    return decorator
