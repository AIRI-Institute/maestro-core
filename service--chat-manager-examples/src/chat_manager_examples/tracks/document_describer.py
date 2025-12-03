from mmar_mapi import Chat, FileStorage, HumanMessage
from mmar_mapi.services import DocumentExtractorAPI, DocExtractionOutput
from mmar_mapi.tracks import SimpleTrack, TrackResponse

from chat_manager_examples.config import DOMAINS


class DocumentDescriber(SimpleTrack):
    DOMAIN = DOMAINS.examples
    CAPTION = "📄🧐 Document Describer"

    def __init__(
        self,
        file_storage: FileStorage,
        document_extractor: DocumentExtractorAPI,
    ):
        self.file_storage = file_storage
        self.document_extractor = document_extractor

    def generate_response(self, chat: Chat, user_message: HumanMessage) -> TrackResponse:
        text = user_message.text
        if text.lower() == "exit":
            return "final", "Bye!"

        user_resource_id = user_message.resource_id
        if user_resource_id:
            user_file_path = self.file_storage.get_path(user_resource_id)
            r_ext = user_file_path.suffix
            if r_ext == ".pdf":
                r_content_rid = self.document_extractor.extract(resource_id=user_resource_id)
                if r_content_rid:
                    r_content = self.file_storage.download_text(r_content_rid)
                    extraction_output = DocExtractionOutput.model_validate_json(r_content)
                    extracted_text = extraction_output.text
                    return f"Распознанное содержимое:\n```\n{extracted_text}\n```"
                else:
                    return "Не удалось распознать содержимое"
            else:
                return "Поддерживаются только файлы с расширением pdf"
        else:
            return "Пришлите файл"
