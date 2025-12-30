import os
from functools import cached_property
from typing import Annotated

from mmar_ptag import LogLevelEnum
from mmar_utils import Message, validate_no_underscores
from mmar_mimpl import ResourcesModel
from pydantic import AfterValidator, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DESCRIPTION = "Я информационная система, созданная в научно-исследовательских целях – AI-ассистент."
AUTH = "Для авторизации используйте команду /authorize [ПАРОЛЬ]."
HELPER = f"Я – AI-ассистент. Нажмите /start, чтобы начать новую консультацию.\n\n{AUTH}"
ACCESS_DENIED = f"Доступ ограничен. {AUTH}"


def validate_commands_dict(commands: dict[str, str]) -> dict[str, str]:
    if "start" not in commands:
        raise ValueError(f"Expected commands has `start`, but found: {commands}")
    if "help" in commands:
        raise ValueError(f"Expected commands has no `help`, but found: {commands}")
    return commands


class LoggerConfig(BaseModel):
    name: str = "telegram-b2c"
    level: LogLevelEnum = LogLevelEnum.DEBUG


class TelethonConfig(BaseModel):
    api_id: int = 0
    api_hash: str = ""
    api_bot_token: str = ""
    session_path: str = "/mnt/data/maestro/local/tg/session.session"


class TgApplicationConfig(BaseModel):
    connect_timeout: int = 30
    read_timeout: int = 30
    write_timeout: int = 30
    utc_delta_hours: int = 3
    connection_pool_size: int = 10

    handle: Annotated[str, AfterValidator(validate_no_underscores)]
    token: str

    user_persistence_path: str = "/mnt/data/maestro/local/tg/user_persistence.pkl"


class AuthConfig(BaseModel):
    disabled: bool = False
    tg_password: str = "example_password"
    whitelist_path: str = "/mnt/data/maestro/local/tg/white_list.csv"
    admin_path: str | None = None


class BotConfig(BaseModel):
    commands: Annotated[dict[str, str], AfterValidator(validate_commands_dict)]
    show_end_button: bool = True
    is_dummy_maestro: bool = False
    allowed_files_extensions: list[str] = ["pdf", "dcm"]


class ResourcesConfig(ResourcesModel):
    helper: str = HELPER
    description: str = DESCRIPTION
    short_description: str = DESCRIPTION

    menu_start: str = "Начать сессию"
    menu_help: str = "Вывести информационное сообщение"
    menu_authorize: str = "Авторизоваться"

    end: str = "Всего хорошего!"
    end_button: str = "Завершить сессию"
    end_message: str = "Сессия завершена"

    # auth
    auth_success: str = "Успешная авторизация!"
    auth_failure: str = "Неверный пароль!"
    auth_repeat: str = "Вы уже авторизованы."
    error: str = "Что-то пошло не так. Повторите запрос или попробуйте позднее."
    access_denied: Message = ACCESS_DENIED
    get_access: Message = ""

    @cached_property
    def reject_access(self) -> list[Message]:
        res = [self.access_denied]
        if self.get_access:
            res.append(self.get_access)
        return res


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")
    version: str = "dev"
    files_dir: str | None = "/mnt/data/maestro/files"

    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    tg_application: TgApplicationConfig = Field(default_factory=TgApplicationConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    bot: BotConfig = Field(default_factory=BotConfig)
    telethon: TelethonConfig = Field(default_factory=TelethonConfig)
    res: ResourcesConfig = Field(default_factory=ResourcesConfig)

    addresses__maestro: str = Field(default='http://gateway:7732')
    timeout: int = 120
    error: str = "Сервер не доступен. Повторите запрос позднее."
    headers_extra: dict[str, str] | None = None


def load_config(env_file: str | None = None) -> Config:
    env_file = env_file or os.getenv("ENV_FILE")
    return Config(_env_file=env_file)
