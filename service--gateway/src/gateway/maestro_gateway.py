import asyncio
from functools import partial

from loguru import logger
from mmar_mapi import AIMessage, Chat, ChatMessage, Context, FileStorage, HumanMessage, make_content
from mmar_mapi.models.tracks import DomainInfo, TrackInfo
from mmar_mapi.services import ChatManagerAPI
from mmar_mcli import FileData, MaestroClientI, MessageData
from mmar_mimpl import installed_trace_id
from mmar_utils import Either
from openai import OpenAI

from gateway.chat_storage import ChatStorageAPI
from gateway.config import Config
from gateway.legacy import ChatRequestOld, ChatResponseOld, make_async
from gateway.models import (
    ChatRequest,
    ChatResponse,
    CreateResponse,
    DBChatPreviews,
    ModelInfo,
    ModelsResponse,
)

Error = str


class MaestroGateway(MaestroClientI):
    def __init__(
        self,
        config: Config,
        file_storage: FileStorage,
        chat_storage: ChatStorageAPI,
        chat_manager: ChatManagerAPI,
        oclient: OpenAI,
    ):
        self._chat_manager = chat_manager
        self._oclient = oclient
        self.end_message: str = config.end_message
        self.default_model = config.default_model
        self.hide_models = set(config.hide_models)
        self.file_storage = file_storage
        self.chat_storage = chat_storage

    @make_async
    def get_domains(self, language_code: str, client_id: str) -> list[DomainInfo]:
        domains: list[DomainInfo] = self._chat_manager.get_domains(language_code=language_code, client_id=client_id)
        return domains

    @make_async
    def get_tracks(self, language_code: str, client_id: str) -> list[TrackInfo]:
        tracks: list[TrackInfo] = self._chat_manager.get_tracks(language_code=language_code, client_id=client_id)
        return tracks

    @make_async
    def get_models(self) -> ModelsResponse:
        try:
            models = self._oclient.models.list()
            # todo fix: how to disginguish models suitable for chat?
            model_infos = [
                ModelInfo(model=model.id, caption=model.id)
                for model in models.data
                if "embedd" not in model.id
                if model.id not in self.hide_models
            ]
            default_model = model_infos[0].model if model_infos else self.default_model
            response = ModelsResponse(models=model_infos, default_model=default_model)
            logger.info(f"#get_models() -> {response}")
            return response
        except Exception:
            logger.exception("Failed to load models from OpenAI, fallback to default")
            return ModelsResponse(
                models=[ModelInfo(model=self.default_model, caption=self.default_model)],
                default_model=self.default_model,
            )

    async def get_chat_previews(self, client_id: str, user_id: str) -> DBChatPreviews:
        return await self.chat_storage.load_chat_previews_by_user_id(client_id, user_id)

    async def delete_chat(self, chat_id: str) -> Either[Error, None]:
        err, _ = await self.chat_storage.delete_chat_by_chat_id(chat_id)
        if err:
            return err, None
        return None, None

    async def get_chat(self, chat_id: str) -> Either[str, Chat]:
        return await self.chat_storage.load_chat_by_chat_id(chat_id)

    async def send_message_by_chat_request_old(self, chat_request: ChatRequestOld) -> ChatResponse:
        context = chat_request.context
        await self._touch_chat(context)
        chat_id = context.create_id()
        cr = ChatRequest(chat_id=chat_id, messages=chat_request.messages)
        res = await self.send_message_by_request(cr)
        return res

    async def send_message_by_request(self, chat_request: ChatRequest) -> ChatResponse:
        chat_id = chat_request.chat_id
        err, chat = await self.chat_storage.load_chat_by_chat_id(chat_id)
        if err:
            raise ValueError(f"Failed to load chat {chat_id}")
        assert chat is not None
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
        assert chat is not None
        return await self.send_message_by_chat(chat, msg)

    async def _touch_chat(self, context: Context) -> None:
        if not await self.chat_storage.has_chat(context):
            chat = await self.chat_storage.load_chat(context)
            await self.chat_storage.dump_chat(chat)

    async def create_chat(self, context: Context) -> CreateResponse:
        chat_id = context.create_id()
        chat = await self.chat_storage.load_chat(context)
        await self.chat_storage.dump_chat(chat)
        return CreateResponse(chat_id=chat_id)

    async def send_message_by_context(self, context: Context, msg: HumanMessage) -> ChatResponseOld:
        # todo align short everywhere
        chat = await self.chat_storage.load_chat(context)
        cr = await self.send_message_by_chat(chat, msg)
        res = ChatResponseOld(
            context=context,
            messages=[msg],
            response_messages=cr.response_messages,
        )
        return res

    async def send_message_by_chat(self, chat: Chat, msg: HumanMessage) -> ChatResponse:
        chat_id = chat.context.create_id()
        logger.info(f"Request to chat_manager, context={chat.context}, user message={msg}, chat_id={chat_id}")
        chat.add_message(msg)
        await self.chat_storage.dump_chat(chat)
        trace_id = chat.context.create_trace_id()

        def get_response() -> list[ChatMessage]:
            with installed_trace_id(trace_id):
                return self._chat_manager.get_response(chat=chat)

        messages = await asyncio.to_thread(get_response)
        for message in messages:
            chat.messages.append(message)

        await self.chat_storage.dump_chat(chat)
        response_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        if not response_messages:
            logger.warning("Not found messages to response...")
        # todo: support long polling in the future
        res = ChatResponse(chat_id=chat_id, response_messages=response_messages)
        return res

    # for MaestroClientI
    # todo fix duplication here and in mmar_mcli.maestro_client
    def get_file_storage(self, client_id: str) -> FileStorage:
        return self.file_storage

    # todo fix duplication here and in mmar_mcli.maestro_client
    async def _download_file_data_maybe(self, msg: AIMessage, client_id: str) -> FileData | None:
        resource_id = msg.resource_id
        if not resource_id:
            return None
        logger.info(f"Downloading resource: {resource_id}")
        resource_name = msg.resource_name
        if not resource_name:
            resource_ext = resource_id.split(".")[-1]
            resource_name = f"result.{resource_ext}"
        resource_bytes = await self.download_resource(resource_id, client_id)
        return resource_name, resource_bytes

    async def send_simple(self, context: Context, msg_data: MessageData | str) -> list[MessageData] | None:
        if isinstance(msg_data, str):
            msg_data = msg_data, None
        content, file_data = msg_data

        resource_id = file_data and await self.upload_resource(file_data, context.client_id)
        content = make_content(content=content, resource_id=resource_id)
        msg = HumanMessage(content=content)

        ai_messages = await self.send(context, msg)

        if not ai_messages:
            return None
        download = partial(self._download_file_data_maybe, client_id=context.client_id)
        res = [(ai_msg.content, await download(ai_msg)) for ai_msg in ai_messages]
        return res or []

    async def send(self, context: Context, msg: HumanMessage) -> list[AIMessage] | None:
        response: ChatResponseOld = await self.send_message_by_context(context, msg)
        ai_messages: list[AIMessage] = [
            ai_msg for ai_msg in response.response_messages if isinstance(ai_msg, AIMessage)
        ]
        return ai_messages or None

    async def upload_resource(self, file_data: FileData, client_id: str) -> str | None:
        file_name, file_bytes = file_data
        resourse_id: str = await self.get_file_storage(client_id).upload_async(file_bytes, file_name)
        logger.debug(f"Uploaded resource with name '{file_name}' to '{resourse_id}'")
        return resourse_id

    async def download_resource(self, resource_id: str, client_id: str) -> bytes:
        res: bytes = await self.get_file_storage(client_id).download_async(resource_id)
        return res
