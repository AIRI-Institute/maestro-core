from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from loguru import logger
from mmar_mapi import Chat, Context
from mmar_utils import Either

from src.io_fs import ensure_existing_dir

from .models import DBChatInfoItem, DBChatPreviews


class ChatStorage:
    def __init__(self, logs_dir: str, logs_dir_archived: str):
        self.logs_dir: Path = ensure_existing_dir(logs_dir)
        self.logs_dir_archived: Path = ensure_existing_dir(logs_dir_archived)

    def _find_chats_by_user_id(self, client_id: str, user_id: str) -> Iterable[str]:
        chat_pattern = f"client_{client_id}_user_{user_id}_session_*.json"
        res = [chat_path.stem for chat_path in self.logs_dir.glob(chat_pattern)]
        return res

    def load_chat_previews_by_user_id(self, client_id: str, user_id: str) -> DBChatPreviews:
        chat_ids: Iterable[str] = self._find_chats_by_user_id(client_id, user_id)
        chat_previews: list[DBChatInfoItem] = []
        for chat_id in chat_ids:
            chat_info = self._load_chat_info_by_chat_id(chat_id)
            if chat_info:
                chat_previews.append(chat_info)

        return DBChatPreviews(chat_previews=chat_previews)

    def _load_chat_info_by_chat_id(self, chat_id: str) -> DBChatInfoItem | None:
        err, chat = self.load_chat_by_chat_id(chat_id)
        if err:
            logger.error(err)
            return None

        first_message, first_message_date = None, None
        if len(messages := chat.messages) > 2:
            first_message = messages[2].body
            first_message_date = messages[2].date_time
        chat_info = DBChatInfoItem(
            chat_id=chat_id,
            first_replica=first_message,
            first_replica_date=first_message_date,
            track_id=chat.context.track_id,
        )
        return chat_info

    def get_chat_path(self, chat_id: str) -> Path:
        return self.logs_dir / f"{chat_id}.json"

    def load_chat_by_chat_id(self, chat_id: str) -> Either[str, Chat]:
        chat_path = self.get_chat_path(chat_id)

        if not chat_path.exists():
            if not chat_id.endswith("_clean"):
                chat_id = f"{chat_id}_clean"
                chat_path = self.get_chat_path(chat_id)
                if not chat_path.exists():
                    return f"Chat not found: {chat_id}", None
            elif chat_id.endswith("_clean"):
                chat_id = chat_id.split("_clean")[0]
                chat_path = self.get_chat_path(chat_id)
                if not chat_path.exists():
                    return f"Chat not found: {chat_id}", None
            else:
                return f"Chat not found: {chat_id}", None

        # handle pydantic validation errors
        try:
            chat = Chat.parse(chat_path.read_text())
        except Exception as ex:
            logger.error(f"Failed to parse {chat_path}: {ex}")
            return f"Failed to parse chat: {chat_id}", None
        return None, chat

    def delete_chat_by_chat_id(self, chat_id: str) -> Either[str, None]:
        chat_path = self.get_chat_path(chat_id)
        archived_path = self.logs_dir_archived / f"{chat_id}.json"

        if not chat_path.exists():
            return f"Chat not found: {chat_id}", None

        if archived_path.exists():
            archived_path = self.logs_dir_archived / f"{chat_id}_{int(datetime.now().timestamp())}.json"

        chat_path.replace(archived_path)
        return None, None

    def dump_chat(self, chat: Chat) -> None:
        fpath = self._make_fpath(chat.context)
        chat_json_text = chat.model_dump_json(indent=2)
        fpath.write_text(chat_json_text)

    def has_chat(self, context: Context) -> Chat:
        fpath = self._make_fpath(context)
        return fpath.exists()

    def load_chat(self, context: Context) -> Chat:
        fpath = self._make_fpath(context)
        if not fpath.exists():
            chat = Chat(context=context)
            logger.info(f"New session created, fpath: {fpath}")
        else:
            chat_text = fpath.read_text()
            try:
                chat = Chat.parse(chat_text)
                logger.info(f"Old session loaded, fpath: {fpath}")
            except Exception:
                logger.error(f"Failed to parse chat: {fpath}")
                raise
        return chat

    def _make_fpath(self, context: Context) -> Path:
        chat_id = context.create_id()
        fname = f"{chat_id}.json"
        return self.logs_dir / fname
