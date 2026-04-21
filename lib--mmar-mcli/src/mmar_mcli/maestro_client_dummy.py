import asyncio
from typing import Any

from loguru import logger
from mmar_mapi import AIMessage, Context, DomainInfo, HumanMessage, TrackInfo
from mmar_mapi.models.chat import _get_command, _get_text, make_content
from mmar_utils import try_parse_int

from mmar_mcli.maestro_client import MaestroClientI
from mmar_mcli.models import MessageData, ModelsResponse


class MaestroClientDummy(MaestroClientI):
    def __init__(self, config: Any):
        pass

    async def get_domains(self, language_code: str, client_id: str) -> list[DomainInfo]:
        return []

    async def get_tracks(self, language_code: str, client_id: str) -> list[TrackInfo]:
        return []

    async def get_models(self) -> ModelsResponse:
        return ModelsResponse(models=[], default_model="")

    async def send_simple(self, context: Context, msg_data: MessageData | str) -> list[MessageData]:
        if isinstance(msg_data, str):
            msg_data = msg_data, None
        content, file_data = msg_data

        text = _get_text(content)
        command = _get_command(content)

        if text.lower().startswith("wait"):
            seconds = try_parse_int(text[len("wait") :].strip())
            logger.info(f"Going to wait {seconds} seconds")
            if seconds:
                await asyncio.sleep(seconds)
            return [(f"After waiting {seconds} seconds", None)]

        text_response_lines = [
            f"Your context: {context}",
            f"Your text: {text}",
            f"Your command: {command}",
            f"Your file_data: {file_data and (file_data[0], len(file_data[1]))}",
        ]
        text_response = "\n".join(text_response_lines)
        return [(text_response, None)]

    async def send(self, context: Context, msg: HumanMessage | str) -> list[AIMessage] | None:
        if isinstance(msg, str):
            msg = HumanMessage(text=msg)

        text_response_lines = [
            f"Your context: {context}",
            f"Your text: {msg.text}",
            f"Your command: {msg.command}",
            f"Your resource_id: {msg.resource_id}",
        ]
        text_response = "\n".join(text_response_lines)
        response = AIMessage(content=make_content(text_response))
        return [response]
