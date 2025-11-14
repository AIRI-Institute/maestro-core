from pathlib import Path

from loguru import logger
from mmar_mapi import AIMessage, Chat, Context, FileStorage, HumanMessage
from mmar_mapi.api import ChatManagerAPI
from mmar_utils import Either

from src.chat_storage import ChatStorage
from src.config import Config
from src.legacy import ChatRequestOld, ChatResponseOld, make_async
from src.models import (
    ChatRequest,
    ChatResponse,
    CreateResponse,
    DBChatPreviews,
    DomainsResponse,
    EntrypointInfo,
    EntrypointsResponse,
    TracksResponse,
)

Error = str


class Gateway:
    def __init__(
        self, config: Config, file_storage: FileStorage, chat_storage: ChatStorage, chat_manager: ChatManagerAPI
    ):
        self._chat_manager = chat_manager
        self.end_message: str = config.end_message
        self.default_entrypoint = config.default_entrypoint
        self.file_storage = file_storage
        self.chat_storage = chat_storage

    @make_async
    def get_domains(self, language_code: str, client_id: str) -> DomainsResponse:
        domains = self._chat_manager.get_domains(language_code=language_code, client_id=client_id)
        return DomainsResponse(domains=domains)

    @make_async
    def get_tracks(self, language_code: str, client_id: str) -> TracksResponse:
        tracks = self._chat_manager.get_tracks(language_code=language_code, client_id=client_id)
        return TracksResponse(tracks=tracks)

    @make_async
    def get_entrypoints(self, client_id: str) -> EntrypointsResponse:
        try:
            context = Context(client_id="Gateway", track_id="SystemEntrypoints")
            chat = Chat(context=context, messages=[HumanMessage(content="/start")])
            ri = self._chat_manager.get_response(chat=chat)[0]
            entrypoints_json = ri.content["result"]
            es_response = EntrypointsResponse.model_validate_json(entrypoints_json)
        except Exception:
            logger.exception("Failed to load entrypoints, fallback to default")
            ei = EntrypointInfo.model_validate(self.default_entrypoint)
            es_response = EntrypointsResponse(default_entrypoint_key=ei.entrypoint_key, entrypoints=[ei])
        return es_response

    @make_async
    def get_chat_previews(self, client_id: str, user_id: str) -> DBChatPreviews:
        return self.chat_storage.load_chat_previews_by_user_id(client_id, user_id)

    @make_async
    def delete_chat(self, chat_id: str) -> Either[Error, None]:
        err, _ = self.chat_storage.delete_chat_by_chat_id(chat_id)
        return err, None

    def get_chat_path(self, chat_id: str) -> Path:
        return self.chat_storage.get_chat_path(chat_id)

    @make_async
    def get_chat(self, chat_id: str) -> Either[str, Chat]:
        return self.chat_storage.load_chat_by_chat_id(chat_id)

    async def send_message_by_chat_request_old(self, chat_request: ChatRequestOld) -> ChatResponse:
        context = chat_request.context
        self._touch_chat(context)
        chat_id = context.create_id()
        cr = ChatRequest(chat_id=chat_id, messages=chat_request.messages)
        res = await self.send_message_by_request(cr)
        return res

    async def send_message_by_request(self, chat_request: ChatRequest) -> ChatResponse:
        chat_id = chat_request.chat_id
        err, chat = self.chat_storage.load_chat_by_chat_id(chat_id)
        if err:
            raise ValueError(f"Failed to load chat {chat_id}")
        # context = chat_request.context
        messages = chat_request.messages
        if not len(messages) == 1:
            raise ValueError(f"Sending only one message supported, found: {messages}")
        msg = messages[0]
        if not isinstance(msg, HumanMessage):
            raise ValueError(f"Expected only human message, found: {msg}")
        return await self.send_message_by_chat(chat, msg)

    async def send_message_by_chat_id(self, chat_id: str, msg: HumanMessage) -> ChatResponse:
        err, chat = await self.get_chat(chat_id)
        if err:
            raise ValueError(err)
        return await self.send_message_by_chat(chat, msg)

    def _touch_chat(self, context: Context) -> None:
        if not self.chat_storage.has_chat(context):
            chat = self.chat_storage.load_chat(context)
            self.chat_storage.dump_chat(chat)

    async def create_chat(self, context: Context) -> CreateResponse:
        chat_id = context.create_id()
        chat = self.chat_storage.load_chat(context)
        self.chat_storage.dump_chat(chat)
        return CreateResponse(chat_id=chat_id)

    async def send_message_by_context(self, context: Context, msg: HumanMessage) -> ChatResponseOld:
        # todo align short everywhere
        chat = self.chat_storage.load_chat(context)
        cr = await self.send_message_by_chat(chat, msg)
        res = ChatResponseOld(
            context=context,
            messages=[msg],
            response_messages=cr.response_messages,
        )
        return res

    @make_async
    def send_message_by_chat(self, chat: Chat, msg: HumanMessage) -> ChatResponse:
        chat_id = chat.context.create_id()
        logger.info(f"Request to chat_manager, context={chat.context}, user message={msg}, chat_id={chat_id}")
        chat.add_message(msg)
        self.chat_storage.dump_chat(chat)
        trace_id = chat.context.create_trace_id()

        messages = self._chat_manager.get_response(chat=chat, trace_id=trace_id)
        for message in messages:
            chat.messages.append(message)

        self.chat_storage.dump_chat(chat)
        response_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        if not response_messages:
            logger.warning("Not found messages to response...")
        # todo fix: support long polling in the future
        res = ChatResponse(
            chat_id=chat_id,
            response_messages=response_messages,
        )
        return res
