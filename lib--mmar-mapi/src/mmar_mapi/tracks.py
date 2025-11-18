from abc import ABC, ABCMeta, abstractmethod

from loguru import logger

from mmar_mapi.models.chat import AIMessage, Chat, ChatMessage, Content, HumanMessage
from mmar_mapi.utils_import import load_main_objects

State = str
TrackResponse = AIMessage | list[AIMessage] | tuple[State, Content]
DOMAIN_CAPTION = {"DOMAIN", "CAPTION"}


def check_class_has_attrs(name, classdict, attrs):
    for attr in attrs:
        if attr not in classdict:
            raise TypeError(f"Class '{name}' must define a '{attr}'")


class MustHaveDomainCaptionMeta(ABCMeta):
    def __new__(cls, name, bases, classdict):
        is_abstract = any({getattr(v, "__isabstractmethod__", None) for v in classdict.values()})

        # Check if "KIND" is defined in the class being created
        if not is_abstract:
            check_class_has_attrs(name, classdict, DOMAIN_CAPTION)
        return super().__new__(cls, name, bases, classdict)


class TrackI(ABC, metaclass=MustHaveDomainCaptionMeta):
    @abstractmethod
    def get_response(self, chat: Chat) -> list[ChatMessage]:
        pass


class StateActionPolicyTrack(TrackI):
    def get_response(self, chat: Chat) -> list[ChatMessage]:
        user_message = chat.get_last_user_message()
        if not user_message:
            # assuming here that track is reactive: if last message is not from user, no reaction needed
            logger.warning("Not user message found! Returning empty")
            return []

        state = self.select_state(chat, user_message)
        action = self.select_action(chat, user_message, state=state)
        content = self.generate_response_content(chat, user_message, action=action)

        logger.debug(f"Bot response: '{content}'")
        extra = dict(action=action)
        bot_message = AIMessage(content=content, extra=extra, state=state)

        messages = [bot_message]
        bot_message = messages[-1]
        prev_state = chat.get_last_state()
        cur_state = bot_message.state
        # todo implement moderation
        logger.info(f"State transition: '{prev_state}' -> '{cur_state}', action: '{action}'")
        return messages

    @abstractmethod
    def select_state(self, chat: Chat, user_message: HumanMessage) -> str:
        pass

    @abstractmethod
    def select_action(self, chat: Chat, user_message: HumanMessage, state: str) -> str:
        pass

    @abstractmethod
    def generate_response_content(self, chat: Chat, user_message: HumanMessage, action: str) -> Content:
        pass


class SimpleTrack(TrackI):
    def get_response(self, chat: Chat) -> list[ChatMessage]:
        user_message = chat.get_last_user_message()
        if not user_message:
            # assuming here that track is reactive: if last message is not from user, no reaction needed
            logger.warning("Not user message found! Returning empty")
            return []

        response: TrackResponse = self.generate_response(chat, user_message)
        if isinstance(response, tuple) and len(response) == 2:
            state: str = response[0]
            content: Content = response[1]
            messages = [AIMessage(state=state, content=content)]
        elif isinstance(response, str):
            state, content = "", response
            messages = [AIMessage(state=state, content=content)]
        elif isinstance(response, AIMessage):
            state = response.state
            messages = [response]
        elif isinstance(response, list):
            if response:
                last_msg = response[-1]
                state = last_msg.state if last_msg.is_ai else ""
                messages = response
            else:
                state = ""
                messages = []
        else:
            logger.error(f"Bad response type: {type(response)}: {response}")
            state = ""
            messages = []
        logger.debug(f"Bot response: '{messages}'")

        track_id = chat.context.track_id
        logger.info(f"Track {track_id}: state transition: '{chat.get_last_state()}' -> '{state}'")
        return messages

    @abstractmethod
    def generate_response(self, chat: Chat, user_message: HumanMessage) -> TrackResponse:
        pass


def load_tracks(tracks_module) -> dict[str, TrackI]:
    return load_main_objects(tracks_module, TrackI)
