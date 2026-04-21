from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ContextModel(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chat_id: Mapped[str] = mapped_column(String, unique=True)
    client_id: Mapped[str] = mapped_column(String)
    user_id: Mapped[str] = mapped_column(String)
    session_id: Mapped[str] = mapped_column(String)
    track_id: Mapped[str] = mapped_column(String)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    messages: Mapped[list["MessageModel"]] = relationship(back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx__sessions__client_id__user_id", "client_id", "user_id"),
    )


class MessageModel(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column()
    type: Mapped[str] = mapped_column(String)
    content: Mapped[Any] = mapped_column(JSONB, nullable=False)
    state: Mapped[str] = mapped_column(String, default="", server_default="")
    date_time: Mapped[str] = mapped_column(String)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["ContextModel"] = relationship(back_populates="messages")

    __table_args__ = (
        UniqueConstraint("session_id", "position", name="uq__messages__session_id__position"),
        Index("idx__messages__session_id", "session_id"),
    )


class SqlAlchemyDatabase:
    def __init__(self, url: str, pool_size: int = 10) -> None:
        self.engine = create_async_engine(url, pool_size=pool_size)
        self.session_factory = async_sessionmaker(bind=self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as s:
            yield s

    async def dispose(self) -> None:
        await self.engine.dispose()
