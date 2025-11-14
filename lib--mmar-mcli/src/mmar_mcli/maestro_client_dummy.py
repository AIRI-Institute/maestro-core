import asyncio

from loguru import logger
from mmar_mapi import Context
from mmar_mapi.models.chat import _get_command, _get_text
from mmar_utils import try_parse_int

from mmar_mcli.maestro_client import MaestroClientI
from mmar_mcli.models import MessageData


class MaestroClientDummy(MaestroClientI):
    def __init__(self, config):
        pass

    async def send(self, context: Context, msg_data: MessageData | str) -> list[MessageData]:
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
