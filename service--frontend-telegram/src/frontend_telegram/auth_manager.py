import asyncio
import re
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger
from mmar_utils import Either

from frontend_telegram.config import Config
from frontend_telegram.io_csv import get_values_from_csv, modify_csv, touch_csv, validate_csv_header, add_row_to_csv
from frontend_telegram.io_fs import get_modified_time, is_empty_file
from frontend_telegram.simple_cache import SET_CACHE_ALWAYS_NO, SET_CACHE_ALWAYS_YES, SetCache, SetCacheI

UserIdUsername = tuple[int, str]
USER_ID = "user_id"
USER_USERNAME = "user_username"
HEADER = [USER_ID, USER_USERNAME]
REFRESH_PERIOD = 10
UserId = int
UserLoader = Callable[[str], Awaitable[Either[Exception, UserId]]]
USERNAME_PATTERN = r"^[a-zA-Z][\w\d]{3,30}[a-zA-Z\d]$"


class AuthManager:
    def __init__(self, config: Config):
        users_disabled = config.auth.disabled or not config.auth.whitelist_path
        admins_disabled = not bool(config.auth.admin_path)

        self.users_white = SET_CACHE_ALWAYS_YES if users_disabled else UsersCache(config.auth.whitelist_path)
        self.users_admins = SET_CACHE_ALWAYS_NO if admins_disabled else UsersCache(config.auth.admin_path)

    def is_authorized(self, user: UserIdUsername) -> bool:
        return user in self.users_white

    def is_admin(self, user: UserIdUsername) -> bool:
        return user in self.users_admins


class UsersCache(SetCacheI[UserIdUsername]):
    def __init__(self, csv_path: str):
        csv_path = Path(csv_path)
        if not csv_path.exists() or is_empty_file(csv_path):
            touch_csv(csv_path, HEADER)
        validate_csv_header(csv_path, HEADER)
        self.ids = CsvColumnCache(csv_path, USER_ID)
        self.usernames = CsvColumnCache(csv_path, USER_USERNAME)
        # todo fix: need to specify type that has name
        self.name = f"Users({csv_path})"
        logger.info(f"Initialized users cache: {csv_path}")

    def __contains__(self, user: UserIdUsername):
        user_id, user_username = user
        return user_id in self.ids or user_username in self.usernames

    def refresh_if_needed(self) -> bool:
        return self.ids.refresh_if_needed() and self.usernames.refresh_if_needed()


class CsvColumnCache(SetCache[Any]):
    def __init__(self, csv_path: Path, column_name: str):
        assert csv_path.exists()
        loader = partial(get_values_from_csv, csv_path=csv_path, column_name=column_name)
        get_mdf_time = partial(get_modified_time, fpath=csv_path)
        super().__init__(name=f"{csv_path}:{column_name}", loader=loader, change_time_getter=get_mdf_time)


async def fill_user_id(row: dict, load_user_id: UserLoader) -> dict | None:
    if row.get(USER_ID):
        return None
    user_username = row[USER_USERNAME]
    err, user_id = await load_user_id(user_username)
    if err:
        logger.warning(f"Failed to get user_id for username={user_username}")
        return None
    return {USER_ID: user_id, USER_USERNAME: user_username}


async def worker_fill_user_id(load_user_id: UserLoader, authlist_path: Path) -> None:
    last_modified = None

    while True:
        modified = get_modified_time(authlist_path)
        if last_modified is None or modified > last_modified:
            last_modified = modified
            # todo fix logic
            # and eliminate, use sqlite
            await modify_csv(authlist_path, partial(fill_user_id, load_user_id=load_user_id))
        await asyncio.sleep(REFRESH_PERIOD)


async def worker_refresh_auth_cache(auth_cache: SetCache) -> None:
    while True:
        changed = auth_cache.refresh_if_needed()
        if changed:
            logger.info(f"Auth cache {auth_cache.name}: refreshed!")
        await asyncio.sleep(REFRESH_PERIOD)


def add_to_authlist(authlist_path: Path, username: str) -> str:
    username = username[1:] if username.startswith("@") else username
    if not authlist_path.exists():
        return "Authlist not exist."
    if not re.match(USERNAME_PATTERN, username):
        return f"Invalid username: `{username}`"
    add_row_to_csv(authlist_path, {USER_ID: "", USER_USERNAME: username})
    authlist_name = authlist_path.stem
    return f"Done! User `{username}` added to {authlist_name}!"
