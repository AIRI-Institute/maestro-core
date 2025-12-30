import re
from collections.abc import Callable
from typing import Any

from loguru import logger
from mmar_mapi import Widget
from mmar_utils import edit_object, flatten
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup, Update
from telegram.ext.filters import COMMAND, PHOTO, TEXT, BaseFilter, Document, MessageFilter

NOTHING = "NOTHING"
# technically user can show button with callback_data = NOTHING, but we assume that is not happen
CMD_PATTERN = f"(?!{NOTHING}$)"


def _make_file_filter_parts(exts: list[str]) -> list[MessageFilter]:
    if not exts:
        return []
    # sugar
    if len(exts) == 1 and exts[0] == "all":
        return [Document.ALL]
    return list(map(Document.FileExtension, exts))


def _union_filter_parts(f_parts: list[MessageFilter]) -> BaseFilter:
    assert f_parts
    if len(f_parts) == 1:
        return f_parts[0]
    head, *tail = f_parts
    res: BaseFilter = head | _union_filter_parts(tail)
    return res


def make_text_or_file_filter(exts: list[str], cmd_allowed: bool = False) -> BaseFilter:
    f_parts = [TEXT, PHOTO] + _make_file_filter_parts(exts)
    res: BaseFilter = _union_filter_parts(f_parts)
    if not cmd_allowed:
        res = res & ~COMMAND
    return res


F_TEXT_PHOTO_PDF = make_text_or_file_filter(exts=["pdf"])
F_TEXT_CMD_PHOTO_PDF = make_text_or_file_filter(exts=["pdf"], cmd_allowed=True)


def get_callbacks(buttons: list[list[InlineKeyboardButton]]) -> list[str]:
    callbacks = [btn.callback_data for btn in flatten(buttons)]
    return callbacks


def btn(data: str, caption: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(caption, callback_data=data)


def make_callback_filter(buttons: list[list[InlineKeyboardButton]]) -> re.Pattern:
    callbacks = get_callbacks(buttons)
    res = re.compile("(" + "|".join(callbacks) + ")")
    return res


def get_user_id_username(update: Update) -> tuple[int, str] | None:
    message = update.message
    if message is None:
        return None
    user = message.from_user
    if user is None:
        return None
    user_id = user.id
    user_username = user.username
    return user_id, user_username


def edit_button(button: object, selected: str, text_callback: Callable) -> InlineKeyboardButton | None:
    if not isinstance(button, InlineKeyboardButton):
        return None
    text_fix = text_callback(button.text) if button.callback_data == selected else button.text
    return InlineKeyboardButton(callback_data=NOTHING, text=text_fix)


def _make_button(text: Any) -> KeyboardButton | None:
    if not isinstance(text, str):
        return None
    return KeyboardButton(text=text)


def make_kbd(buttons: list[list[str]]) -> ReplyKeyboardMarkup:
    tg_buttons = edit_object(buttons, editor=_make_button)
    kbd = ReplyKeyboardMarkup(keyboard=tg_buttons, resize_keyboard=True)
    return kbd


def _make_inline_button(text: Any) -> InlineKeyboardButton | None:
    if not isinstance(text, str):
        return None
    callback_data, text = text.split(":", 1)
    return InlineKeyboardButton(callback_data=callback_data, text=text)


def make_inline_kbd(ibuttons: list[list[str]]) -> InlineKeyboardMarkup:
    tg_ibuttons = edit_object(ibuttons, editor=_make_inline_button)
    return InlineKeyboardMarkup(tg_ibuttons)


def make_kbd_from_w(w: Widget | None) -> InlineKeyboardMarkup | ReplyKeyboardMarkup | None:
    if not w:
        return None
    if w.ibuttons:
        return make_inline_kbd(w.ibuttons)
    if w.buttons:
        # todo fix: is it possible to show buttons and ibuttons simultaneously? 🤔
        # todo fix: is it possible to hide buttons after click?
        return make_kbd(w.buttons)
    logger.warning(f"Unexpected widget with no buttons: {w}")
    return None
