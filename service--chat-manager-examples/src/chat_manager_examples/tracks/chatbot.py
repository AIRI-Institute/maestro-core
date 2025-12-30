from enum import StrEnum

from loguru import logger

from chat_manager_examples.config import DOMAINS
from mmar_mapi import AIMessage, Chat, FileStorage, HumanMessage
from mmar_mapi.services import LLMHubAPI, LLMPayload, Message
from mmar_mapi.tracks import SimpleTrack, TrackResponse

S = StrEnum("State", ["EMPTY", "START", "FINAL"])  # type: ignore[misc]
SYSTEM_PROMPT = "Ты бот-помощник"
SYSTEM_MESSAGE = Message(role="system", content=SYSTEM_PROMPT)


class Chatbot(SimpleTrack):
    DOMAIN = DOMAINS.examples
    CAPTION = "🤖 Chat"

    def __init__(self, file_storage: FileStorage, llm_hub: LLMHubAPI) -> None:
        self.llm_hub = llm_hub
        self.file_storage = file_storage

    def generate_response(self, chat: Chat, user_message: HumanMessage) -> TrackResponse:
        messages = [SYSTEM_MESSAGE, *list(filter(None, map(Message.create, chat.messages)))]
        payload = LLMPayload(messages=messages)
        try:
            response = self.llm_hub.get_response(request=payload)
            logger.info(f"Response from llm-hub: {response}")
            return AIMessage(content=response, state=S.START)
        except Exception:
            logger.exception("Failed to run get_response")
            try:
                keys = self.llm_hub.get_metadata().get_endpoint_keys()
                response = "Not found LLM endpoints, need to configure..." if not keys else "Failed to access llm-hub"
            except Exception:
                logger.error("llm_hub is not accessible: maybe it is not initialized?")
                response = "Failed to access LLM..."
            return AIMessage(content=response, state=S.FINAL)
