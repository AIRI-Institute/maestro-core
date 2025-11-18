import warnings
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, NotRequired, TypeVar

from pydantic import Field
from typing_extensions import TypedDict

from mmar_mapi.models.widget import Widget
from mmar_mapi.type_union import TypeUnion

from .base import Base

_DT_FORMAT: str = "%Y-%m-%d-%H-%M-%S"
_EXAMPLE_DT: str = datetime(year=1970, month=1, day=1).strftime(_DT_FORMAT)
StrDict = dict[str, Any]


class ResourceDict(TypedDict):
    type: Literal["resource_id"]
    resource_id: str
    resource_name: NotRequired[str]


class TextDict(TypedDict):
    type: Literal["text"]
    text: str


class CommandDict(TypedDict):
    type: Literal["command"]
    command: StrDict


ContentBase = str | Widget | ResourceDict | CommandDict | TextDict | StrDict
Content = ContentBase | list[ContentBase]
T = TypeVar("T")


def now_pretty() -> str:
    return datetime.now().strftime(_DT_FORMAT)


class Context(Base):
    client_id: str = Field("", examples=["543216789"])
    user_id: str = Field("", examples=["123456789"])
    session_id: str = Field(default_factory=now_pretty, examples=["987654321"])
    track_id: str = Field("", examples=["Hello"])
    extra: StrDict | None = Field(None, examples=[None])

    def create_id(self, short: bool = False) -> str:
        uid, sid, cid = self.user_id, self.session_id, self.client_id
        return f"client_{cid}_user_{uid}_session_{sid}"

    def create_trace_id(self) -> str:
        uid, sid, cid = self.user_id, self.session_id, self.client_id
        return f"{cid}_{uid}_{sid}"

    def _get_deprecated_extra(self, field, default):
        # legacy: eliminate after migration
        res = (self.extra or {}).get(field, default)
        warnings.warn(f"Deprecated property `{field}`, should be eliminated", stacklevel=2)
        return res

    # fmt: off
    @property
    def sex(self) -> bool: return self._get_deprecated_extra('sex', True)
    @property
    def age(self) -> int: return self._get_deprecated_extra('age', 0)
    @property
    def entrypoint_key(self) -> str: return self._get_deprecated_extra('entrypoint_key', '')
    @property
    def language_code(self) -> str: return self._get_deprecated_extra('language_code', '')
    @property
    def parent_session_id(self) -> str: return self._get_deprecated_extra('parent_session_id', '')
    # fmt: on


def _get_field(obj: Content, field, val_type: type[T]) -> T | None:
    if not isinstance(obj, dict):
        return None
    val = obj.get(field)
    if val is not None and isinstance(val, val_type):
        return val
    return None


def _get_text(obj: Content) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return "".join(map(_get_text, obj))
    if isinstance(obj, dict) and obj.get("type") == "text":
        return _get_field(obj, "text", str) or ""
    return ""


def _modify_text_base(obj: ContentBase, callback: Callable[[str], str]) -> ContentBase:
    if isinstance(obj, str):
        return callback(obj)
    if isinstance(obj, dict) and obj.get("type") == "text":
        text = _get_field(obj, "text", str) or ""
        text_upd = callback(text)
        return {"type": "text", "text": text_upd}
    return obj


def _modify_text(obj: Content, callback: Callable[[str], str]) -> Content:
    if isinstance(obj, list):
        return [_modify_text_base(el, callback) for el in obj]

    return _modify_text_base(obj, callback)


def _get_resource_id(obj: Content) -> str | None:
    if isinstance(obj, list):
        return next((el for el in map(_get_resource_id, obj) if el), None)
    if isinstance(obj, dict) and obj.get("type") == "resource_id":
        return _get_field(obj, "resource_id", str)
    return None


def _get_resource_name(obj: Content) -> str | None:
    if isinstance(obj, list):
        return next((el for el in map(_get_resource_name, obj) if el), None)
    if isinstance(obj, dict) and obj.get("type") == "resource_id":
        return _get_field(obj, "resource_name", str)
    return None


def _get_resource(obj: Content) -> ResourceDict | None:
    if isinstance(obj, list):
        return next((el for el in map(_get_resource_id, obj) if el), None)
    if isinstance(obj, dict) and obj.get("type") == "resource_id":
        return obj
    return None


def _get_command(obj: Content) -> dict | None:
    if isinstance(obj, list):
        return next((el for el in map(_get_command, obj) if el), None)
    if isinstance(obj, dict) and obj.get("type") == "command":
        return _get_field(obj, "command", dict)
    return None


def _get_widget(obj: Content) -> Widget | None:
    if isinstance(obj, list):
        return next((el for el in map(_get_widget, obj) if el), None)
    if isinstance(obj, Widget):
        return obj
    return None


class BaseMessage(Base):
    type: str
    content: Content = Field("", examples=["Привет"])
    date_time: str = Field(default_factory=now_pretty, examples=[_EXAMPLE_DT])
    extra: StrDict | None = Field(None, examples=[None])

    @property
    def text(self) -> str:
        return _get_text(self.content)

    def modify_text(self, callback: Callable[[str], str]) -> "BaseMessage":
        content_upd = _modify_text(self.content, callback)
        return self.with_content(content_upd)

    def with_content(self, content: Content) -> "BaseMessage":
        return self.model_copy(update=dict(content=content))

    @property
    def resource_id(self) -> str | None:
        return _get_resource_id(self.content)

    @property
    def resource_name(self) -> str | None:
        res = _get_resource_name(self.content)
        return res

    @property
    def resource(self) -> dict | None:
        return _get_resource(self.content)

    @property
    def command(self) -> dict | None:
        return _get_command(self.content)

    @property
    def widget(self) -> Widget | None:
        return _get_widget(self.content)

    def with_now_datetime(self):
        return self.model_copy(update=dict(date_time=now_pretty()))

    @property
    def is_ai(self):
        return self.type == "ai"

    @property
    def is_human(self):
        return self.type == "human"

    @staticmethod
    def DATETIME_FORMAT() -> str:
        return _DT_FORMAT

    @staticmethod
    def find_resource_id(msg: "BaseMessage", ext: str | None = None, type: str | None = None) -> str | None:
        resource_id = msg.resource_id
        if type and type != msg.type:
            return None
        if not resource_id:
            return None
        if ext and not resource_id.endswith(ext):
            return None
        return resource_id

    @staticmethod
    def has_state(msg: "BaseMessage", state: str) -> Any | None:
        if not msg.is_ai:
            return None
        return msg if msg.state == state else None


class HumanMessage(BaseMessage):
    type: Literal["human"] = "human"


class AIMessage(BaseMessage):
    type: Literal["ai"] = "ai"
    state: str = Field("", examples=["COLLECTION"])

    @property
    def action(self) -> str:
        return (self.extra or {}).get("action", "")

    def with_state(self, state: str) -> "AIMessage":
        return self.model_copy(update=dict(state=state))


class MiscMessage(BaseMessage):
    type: Literal["misc"] = "misc"


ChatMessage = TypeUnion[HumanMessage, AIMessage, MiscMessage]


def find_in_messages(messages: list[ChatMessage], func: Callable[[ChatMessage], T | None]) -> T | None:
    return next(filter(None, map(func, messages)), None)


class Chat(Base):
    context: Context = Field(default_factory=Context)
    messages: list[ChatMessage] = Field(default_factory=list)

    model_config = {"extra": "ignore"}

    def __init__(self, **data):
        extra_fields = set(data.keys()) - set(type(self).model_fields.keys())
        if extra_fields:
            warnings.warn(f"Chat initialization: extra fields will be ignored: {extra_fields}")
        super().__init__(**data)

    def create_id(self, short: bool = False) -> str:
        return self.context.create_id(short)

    @staticmethod
    def parse(chat_obj: str | dict) -> "Chat":
        return _parse_chat(chat_obj)

    def add_message(self, message: ChatMessage):
        self.messages.append(message)

    def add_messages(self, messages: list[ChatMessage]):
        for message in messages:
            self.messages.append(message)

    def replace_messages(self, messages: list[ChatMessage]):
        return self.model_copy(update=dict(messages=messages))

    def get_last_state(self, default: str = "empty") -> str:
        for ii in range(len(self.messages) - 1, -1, -1):
            message = self.messages[ii]
            if isinstance(message, AIMessage):
                return message.state
        return default

    def find_in_messages(self, func: Callable[[ChatMessage], T | None]) -> T | None:
        return find_in_messages(self.messages, func)

    def rfind_in_messages(self, func: Callable[[ChatMessage], T | None]) -> T | None:
        return find_in_messages(self.messages[::-1], func)

    def get_last_user_message(self) -> HumanMessage | None:
        messages = self.messages
        if not messages:
            return None
        message = messages[-1]
        return message if isinstance(message, HumanMessage) else None

    def count_messages(self, func: Callable[[ChatMessage], bool] | type) -> int:
        if isinstance(func, type):
            msg_type = func
            func = lambda msg: isinstance(msg, msg_type)  # noqa: E731
        return sum(map(func, self.messages))


def make_content(
    text: str | None = None,
    *,
    resource_id: str | None = None,
    resource: dict | None = None,
    command: dict | None = None,
    widget: Widget | None = None,
    content: Content | None = None,
) -> Content:
    if resource and resource_id:
        raise ValueError("Cannot pass both 'resource' and 'resource_id'")

    if resource_id:
        resource = {"type": "resource_id", "resource_id": resource_id}
    elif resource:
        if not isinstance(resource, dict):
            raise TypeError("'resource' must be a dict")
        resource_id = resource.get("resource_id")
        if not resource_id:
            raise ValueError("'resource' must contain 'resource_id'")
        resource_name = resource.get("resource_name")
        resource = {"type": "resource_id", "resource_id": resource_id}
        if resource_name:
            resource["resource_name"] = resource_name
    else:
        resource = None

    command = (command or None) and {"type": "command", "command": command}

    content = content if isinstance(content, list) else [content] if content else []
    content += list(filter(None, [text, resource, command, widget]))
    if len(content) == 0:
        content = ""
    elif len(content) == 1:
        content = content[0]
    return content


def _parse_chat(chat_obj: str | dict | Chat) -> Chat:
    if isinstance(chat_obj, Chat):
        return chat_obj
    if isinstance(chat_obj, dict):
        return Chat.model_validate(chat_obj)
    if isinstance(chat_obj, str):
        return Chat.model_validate_json(chat_obj)
    raise ValueError(f"Bad chat_obj {type(chat_obj)}: {chat_obj}")
