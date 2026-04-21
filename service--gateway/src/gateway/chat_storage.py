from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from loguru import logger
from mmar_mapi import Chat, Context
from mmar_utils import Either

from gateway.io_fs import ensure_existing_dir

from .models import DBChatInfoItem, DBChatPreviews


class ChatStorageAPI(ABC):
    @abstractmethod
    async def load_chat(self, context: Context) -> Chat: ...

    @abstractmethod
    async def load_chat_by_chat_id(self, chat_id: str) -> Either[str, Chat]: ...

    @abstractmethod
    async def dump_chat(self, chat: Chat) -> None: ...

    @abstractmethod
    async def has_chat(self, context: Context) -> bool: ...

    @abstractmethod
    async def delete_chat_by_chat_id(self, chat_id: str) -> Either[str, None]: ...

    @abstractmethod
    async def load_chat_previews_by_user_id(self, client_id: str, user_id: str) -> DBChatPreviews: ...


class ChatStorageFS(ChatStorageAPI):
    def __init__(self, logs_dir: str, logs_dir_archived: str | None = None):
        self.logs_dir: Path = ensure_existing_dir(logs_dir)
        self.logs_dir_archived: Path | None = ensure_existing_dir(logs_dir_archived) if logs_dir_archived else None

    def _make_fpath(self, context: Context) -> Path:
        return self.logs_dir / f"{context.create_id()}.json"

    def _get_chat_path(self, chat_id: str) -> Path:
        return self.logs_dir / f"{chat_id}.json"

    async def load_chat(self, context: Context) -> Chat:
        fpath = self._make_fpath(context)
        if not fpath.exists():
            logger.info(f"New session created, fpath: {fpath}")
            return Chat(context=context)
        try:
            chat = Chat.parse(fpath.read_text())
            logger.info(f"Old session loaded, fpath: {fpath}")
        except Exception:
            logger.error(f"Failed to parse chat: {fpath}")
            raise
        return chat

    async def dump_chat(self, chat: Chat) -> None:
        fpath = self._make_fpath(chat.context)
        fpath.write_text(chat.model_dump_json(indent=2))

    async def has_chat(self, context: Context) -> bool:
        return self._make_fpath(context).exists()

    async def load_chat_by_chat_id(self, chat_id: str) -> Either[str, Chat]:
        chat_path = self._get_chat_path(chat_id)

        if not chat_path.exists():
            if not chat_id.endswith("_clean"):
                chat_id = f"{chat_id}_clean"
                chat_path = self._get_chat_path(chat_id)
                if not chat_path.exists():
                    return f"Chat not found: {chat_id}", None
            elif chat_id.endswith("_clean"):
                chat_id = chat_id.split("_clean")[0]
                chat_path = self._get_chat_path(chat_id)
                if not chat_path.exists():
                    return f"Chat not found: {chat_id}", None
            else:
                return f"Chat not found: {chat_id}", None

        try:
            chat = Chat.parse(chat_path.read_text())
        except Exception as ex:
            logger.error(f"Failed to parse {chat_path}: {ex}")
            return f"Failed to parse chat: {chat_id}", None
        return None, chat

    async def delete_chat_by_chat_id(self, chat_id: str) -> Either[str, None]:
        if self.logs_dir_archived is None:
            raise NotImplementedError("delete_chat is not supported: archive_dir is not configured")

        chat_path = self._get_chat_path(chat_id)
        archived_path = self.logs_dir_archived / f"{chat_id}.json"

        if not chat_path.exists():
            return f"Chat not found: {chat_id}", None

        if archived_path.exists():
            archived_path = self.logs_dir_archived / f"{chat_id}_{int(datetime.now().timestamp())}.json"

        chat_path.replace(archived_path)
        return None, None

    async def load_chat_previews_by_user_id(self, client_id: str, user_id: str) -> DBChatPreviews:
        chat_ids = [p.stem for p in self.logs_dir.glob(f"client_{client_id}_user_{user_id}_session_*.json")]
        chat_previews: list[DBChatInfoItem] = []
        for chat_id in chat_ids:
            err, chat = await self.load_chat_by_chat_id(chat_id)
            if err:
                logger.error(f"Failed to load chat with chat_id={chat_id}: {err}")
                continue
            assert chat is not None
            first_message, first_message_date = None, None
            if len(messages := chat.messages) > 2:
                first_message = messages[2].text
                first_message_date = messages[2].date_time
            chat_previews.append(
                DBChatInfoItem(
                    chat_id=chat_id,
                    first_replica=first_message,
                    first_replica_date=first_message_date,
                    track_id=chat.context.track_id,
                )
            )
        return DBChatPreviews(chat_previews=chat_previews)
