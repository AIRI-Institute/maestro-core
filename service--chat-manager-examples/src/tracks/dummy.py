from mmar_mapi import AIMessage, Chat, FileStorage, HumanMessage, make_content
from mmar_mapi.tracks import SimpleTrack, TrackResponse

from src.config import DOMAINS


class Dummy(SimpleTrack):
    DOMAIN = DOMAINS.examples
    CAPTION = "🛠 Dummy"

    def __init__(self, config):
        self.file_storage = FileStorage(config.files_dir)

    def generate_response(self, chat: Chat, user_message: HumanMessage) -> TrackResponse:
        text = user_message.text
        if text.lower() == "exit":
            return AIMessage(content="Exit!", state="final")
        resource_id = user_message.resource_id

        if text:
            response_text_info = f"Сообщение: '{text}'"
        else:
            response_text_info = "Сообщение: нет"

        if resource_id:
            resource_name = self.file_storage.get_fname(resource_id)
            response_resource = dict(resource_id=resource_id, resource_name=f"response-{resource_name}")
            response_resource_info = f"файл: '{self.file_storage.get_fname(resource_id)}'."
        else:
            response_resource = None
            response_resource_info = "файл: нет"

        response_text = ", ".join([response_text_info, response_resource_info])
        content = make_content(text=response_text, resource=response_resource)
        msg = AIMessage(content=content, state="dummy")
        return msg
