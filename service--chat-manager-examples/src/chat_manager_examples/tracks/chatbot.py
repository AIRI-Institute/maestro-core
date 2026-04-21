from collections.abc import Iterable

from loguru import logger
from mmar_mapi import AIMessage, Chat, HumanMessage
from mmar_mapi.tracks import SimpleTrack, TrackResponse
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from chat_manager_examples.config import DOMAINS

SYSTEM_PROMPT = "Ты бот-помощник"


class Chatbot(SimpleTrack):
    DOMAIN = DOMAINS.examples
    CAPTION = "🤖 Chat"

    def __init__(self, oclient: OpenAI) -> None:
        self.oclient = oclient

    def _build_messages(self, chat: Chat) -> Iterable[ChatCompletionMessageParam]:
        """Build messages array with system prompt and conversation history."""
        messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in chat.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.text})
            elif isinstance(msg, AIMessage) and msg.text:
                messages.append({"role": "assistant", "content": msg.text})

        return messages

    def generate_response(self, chat: Chat, user_message: HumanMessage) -> TrackResponse:
        messages = self._build_messages(chat)

        try:
            response = self.oclient.chat.completions.create(
                model=chat.context.model,
                messages=messages,
            )
            response_text = response.choices[0].message.content or ""
            logger.info(f"Response from LLM: {response_text[:100]}...")
            return "start", response_text
        except Exception:
            logger.exception("Failed to get LLM response")
            return "final", "Failed to access LLM..."
