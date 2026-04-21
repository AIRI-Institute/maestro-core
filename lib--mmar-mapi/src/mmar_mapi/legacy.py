import json
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from mmar_mapi import AIMessage, Chat, ChatMessage, Context, HumanMessage, make_content
from mmar_mapi.models.widget import Widget

_DT_FORMAT: str = "%Y-%m-%d-%H-%M-%S"
_EXAMPLE_DT_0 = datetime(1970, 1, 1, 0, 0, 0)
_EXAMPLE_DT: str = _EXAMPLE_DT_0.strftime(_DT_FORMAT)


class Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    @model_validator(mode="before")
    @classmethod
    def validate_to_json(cls, value: str | Any) -> Any:
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


def now_pretty() -> str:
    return datetime.now().strftime(ReplicaItem.DATETIME_FORMAT())


class OuterContextItem(BaseModel):
    # remove annoying warning for protected `model_` namespace
    model_config = ConfigDict(protected_namespaces=())

    sex: bool = Field(False, alias="Sex", description="True = male, False = female", examples=[True])
    age: int = Field(0, alias="Age", examples=[20])
    user_id: str = Field("", alias="UserId", examples=["123456789"])
    parent_session_id: str | None = Field(None, alias="ParentSessionId", examples=["987654320"])
    session_id: str = Field("", alias="SessionId", examples=["987654321"])
    client_id: str = Field("", alias="ClientId", examples=["543216789"])
    track_id: str = Field(default="Consultation", alias="TrackId")
    entrypoint_key: str = Field("", alias="EntrypointKey", examples=["giga"])
    language_code: str = Field("ru", alias="LanguageCode", examples=["ru"])

    def create_id(self, short: bool = False) -> str:
        uid, sid, cid = self.user_id, self.session_id, self.client_id
        if short:
            return f"{uid}_{sid}_{cid}"
        return f"user_{uid}_session_{sid}_client_{cid}"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)


def nullify_empty(text: str) -> str | None:
    return text or None


class ReplicaItem(Base):
    body: str = Field("", alias="Body", examples=["Привет"])
    resource_id: Annotated[str | None, AfterValidator(nullify_empty)] = Field(
        None, alias="ResourceId", examples=["<link-id>"]
    )
    resource_name: Annotated[str | None, AfterValidator(nullify_empty)] = Field(
        None, alias="ResourceName", examples=["filename"]
    )
    widget: Widget | None = Field(None, alias="Widget", examples=[None])
    command: dict | None = Field(None, alias="Command", examples=[None])
    role: bool = Field(False, alias="Role", description="True = ai, False = client", examples=[False])
    date_time: str = Field(
        default_factory=now_pretty, alias="DateTime", examples=[_EXAMPLE_DT], description=f"Format: {_DT_FORMAT}"
    )
    state: str = Field("", alias="State", description="chat manager fsm state", examples=["COLLECTION"])
    action: str = Field("", alias="Action", description="chat manager fsm action", examples=["DIAGNOSIS"])
    # todo fix: support loading from `moderation: int`
    moderation: Annotated[str, BeforeValidator(str)] = Field(
        "OK", alias="Moderation", description="moderation outcome", examples=["OK"]
    )
    extra: dict | None = Field(None, alias="Extra", examples=[None])

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)

    @staticmethod
    def DATETIME_FORMAT() -> str:
        return _DT_FORMAT

    def with_now_datetime(self):
        return self.model_copy(update=dict(date_time=now_pretty()))

    @property
    def is_ai(self):
        return self.role

    @property
    def is_human(self):
        return not self.role

    def modify_text(self, callback: Callable[[str], str]) -> "ReplicaItem":
        body_upd = callback(self.body)
        return self.model_copy(update=dict(body=body_upd))


class InnerContextItem(Base):
    replicas: list[ReplicaItem] = Field(alias="Replicas")
    attrs: dict[str, str | int] | None = Field(default={}, alias="Attrs")

    def to_dict(self) -> dict[str, list]:
        return self.model_dump(by_alias=True)


class ChatItem(Base):
    outer_context: OuterContextItem = Field(alias="OuterContext")
    inner_context: InnerContextItem = Field(alias="InnerContext")

    def create_id(self, short: bool = False) -> str:
        return self.outer_context.create_id(short)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)

    def add_replica(self, replica: ReplicaItem):
        self.inner_context.replicas.append(replica)

    def add_replicas(self, replicas: list[ReplicaItem]):
        for replica in replicas:
            self.inner_context.replicas.append(replica)

    def replace_replicas(self, replicas: list[ReplicaItem]):
        return self.model_copy(update=dict(inner_context=InnerContextItem(replicas=replicas)))

    def get_last_state(self, default: str = "empty") -> str:
        replicas = self.inner_context.replicas
        for ii in range(len(replicas) - 1, -1, -1):
            replica = replicas[ii]
            if replica.role:
                return replica.state
        return default

    def zip_history(self, field: str) -> list[Any]:
        return [replica.to_dict().get(field, None) for replica in self.inner_context.replicas]

    @classmethod
    def parse(cls, chat_obj: str | dict) -> "ChatItem":
        return _parse_chat_item(chat_obj)


def _parse_chat_item(chat_obj: str | dict) -> ChatItem:
    if isinstance(chat_obj, dict):
        return ChatItem.model_validate(chat_obj)

    return ChatItem.model_validate_json(chat_obj)


def convert_replica_item_to_message(replica: ReplicaItem) -> ChatMessage:
    date_time = replica.date_time
    content = make_content(
        text=replica.body,
        resource_id=replica.resource_id,
        command=replica.command,
        widget=replica.widget,
    )
    # legacy: eliminate after migration
    if resource_id := replica.resource_id:
        resource = {"type": "resource_id", "resource_id": resource_id}
        resource_name = replica.resource_name
        if resource_name:
            resource["resource_name"] = resource_name
    else:
        resource = None
    body = replica.body
    command = (replica.command or None) and {"type": "command", "command": replica.command}
    widget = replica.widget
    date_time = replica.date_time

    content = list(filter(None, [body, resource, command, widget]))
    if len(content) == 0:
        content = ""
    elif len(content) == 1:
        content = content[0]

    is_bot_message = replica.role

    if is_bot_message:
        kwargs = dict(
            content=content,
            date_time=date_time,
            state=replica.state,
            extra=dict(
                **(replica.extra or {}),
                action=replica.action,
                moderation=replica.moderation,
            ),
        )
        res = AIMessage(**kwargs)
    else:
        kwargs = dict(content=content, date_time=date_time)
        res = HumanMessage(**kwargs)
    return res


def convert_outer_context_to_context(octx: OuterContextItem) -> Context:
    # legacy: eliminate after migration
    context = Context(
        client_id=octx.client_id,
        user_id=octx.user_id,
        session_id=octx.session_id,
        track_id=octx.track_id,
        extra=dict(
            sex=octx.sex,
            age=octx.age,
            parent_session_id=octx.parent_session_id,
            entrypoint_key=octx.entrypoint_key,
            language_code=octx.language_code,
        ),
    )
    return context


def convert_chat_item_to_chat(chat_item: ChatItem) -> Chat:
    # legacy: eliminate after migration
    context = convert_outer_context_to_context(chat_item.outer_context)
    messages = list(map(convert_replica_item_to_message, chat_item.inner_context.replicas))
    res = Chat(context=context, messages=messages)
    return res


def convert_context_to_outer_context(context: Context, failsafe: bool = False) -> OuterContextItem:
    # legacy: eliminate after migration
    extra = context.extra or {}
    if failsafe:
        extra["sex"] = extra.get("sex") or True
        extra["age"] = extra.get("age") or 42
        extra["language_code"] = extra.get("language_code") or ""
        extra["entrypoint_key"] = extra.get("entrypoint_key") or ""
    return OuterContextItem(
        ClientId=context.client_id,
        UserId=context.user_id,
        SessionId=context.session_id,
        TrackId=context.track_id,
        Sex=extra["sex"],
        Age=extra["age"],
        EntrypointKey=extra["entrypoint_key"],
        LanguageCode=extra["language_code"],
        ParentSessionId=extra.get("parent_session_id"),
    )


def convert_message_to_replica_item(message: ChatMessage) -> ReplicaItem | None:
    # legacy: eliminate after migration
    m_type = message.type
    if m_type in {"ai", "human"}:
        role = m_type == "ai"
    else:
        return None

    extra = deepcopy(message.extra) if message.extra else {}
    action = extra.pop("action", "")
    moderation = extra.pop("moderation", "OK")

    kwargs = dict(
        role=role,
        body=message.text,
        resource_id=message.resource_id,
        resource_name=message.resource_name,
        command=message.command,
        widget=message.widget,
        date_time=message.date_time,
        extra=extra or None,
        state=getattr(message, "state", ""),
        action=action,
        moderation=moderation,
    )
    return ReplicaItem(**kwargs)


def convert_chat_to_chat_item(chat: Chat, failsafe: bool = False) -> ChatItem:
    # legacy: eliminate after migration
    res = ChatItem(
        outer_context=convert_context_to_outer_context(chat.context, failsafe=failsafe),
        inner_context=dict(replicas=list(map(convert_message_to_replica_item, chat.messages))),
    )
    return res

def parse_chat_item_as_chat(chat_obj: str | dict | ChatItem) -> Chat:
    # legacy: eliminate after migration
    if isinstance(chat_obj, ChatItem):
        chat_item = chat_obj
    else:
        chat_item = ChatItem.parse(chat_obj)
    res = convert_chat_item_to_chat(chat_item)
    return res


def _parse_chat(chat_obj: str | dict) -> Chat:
    if isinstance(chat_obj, dict):
        return Chat.model_validate(chat_obj)

    return Chat.model_validate_json(chat_obj)


def is_chat_item(chat_obj: str | dict | ChatItem) -> bool:
    if isinstance(chat_obj, ChatItem):
        return True
    if isinstance(chat_obj, dict):
        return "OuterContext" in chat_obj
    if isinstance(chat_obj, str):
        return "OuterContext" in chat_obj
    warnings.warn(f"Unexpected chat object: {chat_obj} :: {type(chat_obj)}")
    return False


def parse_chat_compat(chat_obj: str | dict | ChatItem) -> Chat:
    # legacy: eliminate after migration
    if is_chat_item(chat_obj):
        chat = parse_chat_item_as_chat(chat_obj)
        return chat
    try:
        return _parse_chat(chat_obj)
    except ValidationError as ex:
        warnings.warn(f"Failed to parse chat: {ex}")
        return parse_chat_item_as_chat(chat_obj)


