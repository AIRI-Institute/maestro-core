from typing import cast

from loguru import logger
from mmar_mapi import AIMessage, BaseMessage, Chat, ChatMessage, Context, HumanMessage, MiscMessage
from mmar_utils import Either
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult

from gateway.chat_storage import ChatStorageAPI
from gateway.db import ContextModel, MessageModel, SqlAlchemyDatabase
from gateway.models import DBChatInfoItem, DBChatPreviews

MESSAGE_CLASSES: dict[str, type[BaseMessage]] = {
    "human": HumanMessage,
    "ai": AIMessage,
    "misc": MiscMessage,
}


def _session_to_context(session: ContextModel) -> Context:
    return Context(
        client_id=session.client_id,
        user_id=session.user_id,
        session_id=session.session_id,
        track_id=session.track_id,
        extra=session.extra,
    )


def _message_to_model(msg: ChatMessage, position: int) -> MessageModel:
    dump = msg.model_dump()
    return MessageModel(
        position=position,
        type=dump["type"],
        content=dump["content"],
        state=dump.get("state", ""),
        date_time=dump["date_time"],
        extra=dump.get("extra"),
    )


def _model_to_message(model: MessageModel) -> ChatMessage:
    cls = MESSAGE_CLASSES.get(model.type)
    if cls is None:
        logger.warning(f"Unknown message type '{model.type}', falling back to MiscMessage")
        cls = MiscMessage
    data = {
        "type": model.type,
        "content": model.content,
        "date_time": model.date_time,
        "extra": model.extra,
    }
    if model.type == "ai":
        data["state"] = model.state
    return cast(ChatMessage, cls.model_validate(data))


class ChatStorageSQL(ChatStorageAPI):
    def __init__(self, db: SqlAlchemyDatabase) -> None:
        self._db = db

    async def load_chat(self, context: Context) -> Chat:
        chat_id = context.create_id()
        async with self._db.session() as session:
            row = await session.execute(select(ContextModel).where(ContextModel.chat_id == chat_id))
            db_session = row.scalar_one_or_none()

            if db_session is None:
                logger.info(f"New session created, chat_id: {chat_id}")
                return Chat(context=context)

            messages_rows = await session.execute(
                select(MessageModel).where(MessageModel.session_id == db_session.id).order_by(MessageModel.position)
            )
            messages = [_model_to_message(m) for m in messages_rows.scalars()]
            logger.info(f"Old session loaded, chat_id: {chat_id}")
            return Chat(context=_session_to_context(db_session), messages=messages)

    async def load_chat_by_chat_id(self, chat_id: str) -> Either[str, Chat]:
        async with self._db.session() as session:
            row = await session.execute(select(ContextModel).where(ContextModel.chat_id == chat_id))
            db_session = row.scalar_one_or_none()

            if db_session is None:
                return f"Chat not found: {chat_id}", None

            messages_rows = await session.execute(
                select(MessageModel).where(MessageModel.session_id == db_session.id).order_by(MessageModel.position)
            )
            messages = [_model_to_message(m) for m in messages_rows.scalars()]
            chat = Chat(context=_session_to_context(db_session), messages=messages)
            return None, chat

    async def dump_chat(self, chat: Chat) -> None:
        chat_id = chat.context.create_id()
        async with self._db.session() as session:
            stmt = (
                pg_insert(ContextModel)
                .values(
                    chat_id=chat_id,
                    client_id=chat.context.client_id,
                    user_id=chat.context.user_id,
                    session_id=chat.context.session_id,
                    track_id=chat.context.track_id,
                    extra=chat.context.extra,
                )
                .on_conflict_do_update(
                    index_elements=["chat_id"],
                    set_={"track_id": chat.context.track_id, "extra": chat.context.extra, "updated_at": func.now()},
                )
                .returning(ContextModel.id)
            )
            result = await session.execute(stmt)
            db_session_id = result.scalar_one()

            existing = await session.execute(
                select(MessageModel.position).where(MessageModel.session_id == db_session_id)
            )
            existing_positions = set(existing.scalars())

            new_messages = []
            for i, msg in enumerate(chat.messages):
                if i not in existing_positions:
                    model = _message_to_model(msg, position=i)
                    model.session_id = db_session_id
                    new_messages.append(model)

            if new_messages:
                session.add_all(new_messages)

            await session.commit()

    async def has_chat(self, context: Context) -> bool:
        chat_id = context.create_id()
        async with self._db.session() as session:
            row = await session.execute(select(ContextModel.id).where(ContextModel.chat_id == chat_id))
            return row.scalar_one_or_none() is not None

    async def delete_chat_by_chat_id(self, chat_id: str) -> Either[str, None]:
        async with self._db.session() as session:
            result = await session.execute(delete(ContextModel).where(ContextModel.chat_id == chat_id))
            await session.commit()
            cursor_result = cast(CursorResult, result)
            if cursor_result.rowcount == 0:
                return f"Chat not found: {chat_id}", None
            return None, None

    # TODO: optimize N+1 queries — consider JOIN or selectinload for messages
    async def load_chat_previews_by_user_id(self, client_id: str, user_id: str) -> DBChatPreviews:
        async with self._db.session() as session:
            rows = await session.execute(
                select(ContextModel).where(
                    ContextModel.client_id == client_id,
                    ContextModel.user_id == user_id,
                )
            )
            db_sessions = rows.scalars().all()

            chat_previews: list[DBChatInfoItem] = []
            for db_session in db_sessions:
                try:
                    messages_rows = await session.execute(
                        select(MessageModel)
                        .where(MessageModel.session_id == db_session.id)
                        .order_by(MessageModel.position)
                        .limit(3)
                    )
                    messages = messages_rows.scalars().all()

                    first_message, first_message_date = None, None
                    if len(messages) > 2:
                        msg = _model_to_message(messages[2])
                        first_message = msg.text
                        first_message_date = messages[2].date_time

                    chat_previews.append(
                        DBChatInfoItem(
                            chat_id=db_session.chat_id,
                            first_replica=first_message,
                            first_replica_date=first_message_date,
                            track_id=db_session.track_id,
                        )
                    )
                except Exception:
                    logger.exception(f"Failed to load preview for chat_id={db_session.chat_id}")

            return DBChatPreviews(chat_previews=chat_previews)
