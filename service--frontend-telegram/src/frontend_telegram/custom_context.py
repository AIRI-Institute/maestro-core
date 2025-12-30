from mmar_mapi import Context
from pydantic import BaseModel
from telegram.ext import CallbackContext, ExtBot


class UserData(BaseModel):
    chat_context: Context | None = None

    # authorization relevant data
    # todo remove later: after authing via password we can add user to whitelist
    is_authorized: bool = False
    prev_message_id: int | None = None

    def get_prev_message_id(self) -> int | None:
        try:
            return self.prev_message_id
        except AttributeError:
            # compat
            return getattr(self, "previous_assistant_message_id")


class ChatData(dict):
    pass


class BotData(dict):
    pass


class AssistantContext(CallbackContext[ExtBot, UserData, ChatData, BotData]):
    pass
