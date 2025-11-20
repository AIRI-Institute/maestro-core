from pathlib import Path

from loguru import logger
from mmar_mapi import Chat, FileStorage, HumanMessage
from mmar_mapi.api import BinaryClassifiersAPI, LLMAccessorAPI, TextExtractorAPI
from mmar_mapi.tracks import SimpleTrack, TrackResponse
from mmar_ptag import ptag_client
from mmar_utils import pretty_prefix

from src.config import DOMAINS, Config

OUT_PREFIX_SIZE = 4000


def get_pretty_size(size_bytes: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def get_text_safe(fpath: Path) -> str | None:
    try:
        text = fpath.read_text()
        return pretty_prefix(text, OUT_PREFIX_SIZE)
    except Exception:
        return None


def is_textual(fpath, blocksize=4096):
    try:
        with open(fpath, "r") as fin:
            fin.read(blocksize)
        return True
    except Exception:
        return False


class Describer(SimpleTrack):
    DOMAIN = DOMAINS.examples
    CAPTION = "💬🧐 Describer"

    def __init__(self, config: Config):
        self.file_storage = FileStorage(config.files_dir)
        self.llm_accessor = ptag_client(LLMAccessorAPI, config.addresses.llm_accessor)
        self.question_detector = ptag_client(BinaryClassifiersAPI, config.addresses.question_detector)
        self.text_extractor = ptag_client(TextExtractorAPI, config.addresses.text_extractor)

    def generate_response(self, chat: Chat, user_message: HumanMessage) -> TrackResponse:
        text = user_message.text
        if text.lower() == "exit":
            return "final", "Bye!"

        user_text = user_message.text
        if user_text:
            response_text_parts = [f"Сообщение пользователя:\n```\n{user_text}\n```"]
            try:
                is_question = self.question_detector.evaluate(text=user_text)
                is_question_pretty = "да, это вопрос" if is_question else "нет, это не вопрос"
                response_text_parts.append(f"Детектор вопросов: {is_question_pretty}")
            except Exception as ex:
                logger.warning(f"question-detector is not available: {ex}")
                response_text_parts.append(f"Детектор вопросов: временно недоступен")
            response_text = "\n".join(response_text_parts)
        else:
            response_text = None

        user_resource_id = user_message.resource_id
        if user_resource_id:
            user_file_path = self.file_storage.get_path(user_resource_id)
            r_name = self.file_storage.get_fname(user_resource_id)
            r_size = user_file_path.stat().st_size
            response_resource_id_parts = [
                f"Имя файла пользователя: `{r_name}`",
                f"Размер файла пользователя: `{get_pretty_size(r_size)}`",
            ]
            r_ext = user_file_path.suffix
            if is_textual(user_file_path):
                user_file_text = get_text_safe(user_file_path)
                if user_file_text is not None:
                    response_resource_id_parts.append(f"Содержимое:\n```\n{user_file_text}\n```")
                else:
                    response_resource_id_parts.append("Содержимое файла: невозможно прочитать")
            elif r_ext in {".jpg", ".pdf", ".png"}:
                r_content_rid = self.text_extractor.extract(resource_id=user_resource_id)
                r_content = self.file_storage.download_text(r_content_rid)
                response_resource_id_parts.append(f"Распознанное содержимое:\n```\n{r_content}\n```")

            response_resource_id = "\n".join(response_resource_id_parts)
        else:
            response_resource_id = None

        response = "\n".join(filter(None, [response_text, response_resource_id]))
        return response
