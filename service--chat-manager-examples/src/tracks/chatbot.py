from enum import StrEnum

from loguru import logger
from mmar_mapi import AIMessage, Chat, FileStorage, HumanMessage
from mmar_mapi.api import LLMAccessorAPI, Message, Payload
from mmar_mapi.tracks import SimpleTrack, TrackResponse
from mmar_ptag import ptag_client

from src.config import DOMAINS, Config

S = StrEnum("State", ["EMPTY", "START", "FINAL"])
SYSTEM_PROMPT = "Ты бот-помощник"
SYSTEM_MESSAGE = Message(role="system", content=SYSTEM_PROMPT)


class Chatbot(SimpleTrack):
    DOMAIN = DOMAINS.examples
    CAPTION = "🤖 Chat"

    def __init__(self, config: Config):
        self.file_storage = FileStorage(config.files_dir)
        self.llm_accessor = ptag_client(LLMAccessorAPI, config.addresses.llm_accessor)

    def generate_response(self, chat: Chat, user_message: HumanMessage) -> TrackResponse:
        messages = [SYSTEM_MESSAGE] + list(filter(None, map(Message.create, chat.messages)))
        payload = Payload(messages=messages)
        try:
            response = self.llm_accessor.get_response(request=payload)
            logger.info(f"Response from llm-accessor: {response}")
            return AIMessage(content=response, state=S.START)
        except Exception:
            logger.exception("Failed to run get_response")
            try:
                keys = self.llm_accessor.get_entrypoint_keys()
                if not keys:
                    response = "Not found LLM entrypoints, need to configure..."
                else:
                    response = "Failed to access llm-accessor"
            except Exception:
                logger.error("llm_accessor is not accessible: maybe it is not initialized?")
                response = "Failed to access LLM..."
            return AIMessage(content=response, state=S.FINAL)
