from pathlib import Path

from loguru import logger
from mmar_utils import Either
from telethon.client import TelegramClient
from telethon.tl.functions.users import GetFullUserRequest

from frontend_telegram.config import TelethonConfig


class TelethonClient:
    def __init__(self, config: TelethonConfig):
        self.bot_token = config.api_bot_token
        session_path = Path(config.session_path)
        session_path.parent.mkdir(parents=True, exist_ok=True)
        self.telethon_client = TelegramClient(
            session=session_path,
            api_id=config.api_id,
            api_hash=config.api_hash,
        )

    async def get_user_id_by_username(self, username: str) -> Either[Exception, int]:
        # self.telethon_client.start(bot_token=config.api_bot_token)
        # todo fix: batch loading of user_ids
        try:
            await self.telethon_client.start(bot_token=self.bot_token)
            if not username:
                return ValueError("Empty telegram username"), None
            try:
                response = await self.telethon_client(GetFullUserRequest(username))
                user = response.users[0]
                user_id = user.id
                logger.info(f"User scraped: {username} -> {user_id}")
                return None, user_id
            except Exception as e:
                return e, None
        finally:
            await self.telethon_client.disconnect()
