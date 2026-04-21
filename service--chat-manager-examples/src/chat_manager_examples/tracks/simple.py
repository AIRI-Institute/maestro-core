from loguru import logger
from mmar_mapi import Chat, HumanMessage
from mmar_mapi.tracks import SimpleTrack, TrackResponse
from openai import OpenAI

from chat_manager_examples.config import DOMAINS


class Simple(SimpleTrack):
    DOMAIN = DOMAINS.examples
    CAPTION = "💃️ Simple"

    def __init__(self, oclient: OpenAI):
        self.oclient = oclient

    def generate_response(self, chat: Chat, user_message: HumanMessage) -> TrackResponse:
        last_state = chat.get_last_state()

        # Handle initial states: None, empty string, "/start", or "empty"
        if not last_state or last_state in ("/start", "empty"):
            return "waiting", "Я цифровой ассистент. Задавайте ваши вопросы"

        # Handle waiting state - call LLM
        if last_state == "waiting":
            try:
                response = self.oclient.chat.completions.create(
                    model=chat.context.model,
                    messages=[{"role": "user", "content": user_message.text}],
                )
                response_text = response.choices[0].message.content or ""
                return "waiting", response_text
            except Exception:
                logger.exception("Failed to process request")
                return "error", "Возникла неизвестная ошибка. Пожалуйста, перезагрузите файл."

        return last_state, "Ошибка при обработке запроса."
