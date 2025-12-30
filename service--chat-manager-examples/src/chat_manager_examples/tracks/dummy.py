from chat_manager_examples.config import DOMAINS
from mmar_mapi import AIMessage, Chat, FileStorage, HumanMessage, make_content
from mmar_mapi.tracks import SimpleTrack, TrackResponse


class Dummy(SimpleTrack):
    DOMAIN = DOMAINS.examples
    CAPTION = "🛠 Dummy"

    def __init__(self, file_storage: FileStorage):
        self.file_storage = file_storage

    def generate_response(self, chat: Chat, user_message: HumanMessage) -> TrackResponse:
        text = user_message.text
        if text.lower() == "exit":
            return "final", "Bye!"
        resource_id = user_message.resource_id

        response_text_info = f"Сообщение: '{text}'" if text else "Сообщение: нет"

        if resource_id:
            resource_name = self.file_storage.get_fname(resource_id)
            response_resource = {"resource_id": resource_id, "resource_name": f"response-{resource_name}"}
            response_resource_info = f"файл: '{self.file_storage.get_fname(resource_id)}'."
        else:
            response_resource = None
            response_resource_info = "файл: нет"

        response_text = ", ".join([response_text_info, response_resource_info])
        content = make_content(text=response_text, resource=response_resource)
        return AIMessage(content=content, state="dummy")
