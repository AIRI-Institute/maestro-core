from enum import StrEnum
from typing import Any

from loguru import logger
from mmar_llm import EntrypointConfig, EntrypointsConfig
from mmar_mapi import AIMessage, Chat, HumanMessage, make_content
from mmar_mapi.models.widget import Widget
from mmar_mapi.tracks import SimpleTrack, TrackResponse

from src.config import DOMAINS, Config

STATES = [
    "EMPTY",  # Initial state
    "CHOOSE_PROVIDER",  # Choosing provider type
    "GIGACHAT_BASE_URL",  # Entering GigaChat BASE_URL
    "GIGACHAT_AUTH_METHOD",  # Choosing auth method
    "GIGACHAT_USER",  # Entering username
    "GIGACHAT_PASSWORD",  # Entering password
    "GIGACHAT_CLIENT_ID",  # Entering client ID
    "GIGACHAT_CLIENT_SECRET",  # Entering client secret
    "GIGACHAT_CAPTION",  # Entering caption
    "GIGACHAT_KEY",  # Entering key
    "OPENROUTER_MODEL_ID",  # Entering OpenRouter model ID
    "OPENROUTER_API_KEY",  # Entering API key
    "OPENROUTER_CAPTION",  # Entering caption
    "OPENROUTER_KEY",  # Entering key
    "FINAL_DEFAULT_ENTRYPOINT",  # Setting default entrypoint
    "FINAL",
]
S = StrEnum("State", STATES)

GIGACHAT_KEYS = ['giga', 'giga-max', 'giga-max-2']
GIGACHAT_DEFAULT_KEY = 'giga-max-2'


class EntrypointsWizard(SimpleTrack):
    DOMAIN = DOMAINS.examples
    CAPTION = "🧙‍♂️🤖 Entrypoints Wizard"

    def __init__(self, config: Config):
        self.config = config
        # todo eliminate hotfixes: track should be stateless
        self.last_session_id = None
        self.collected_entrypoints: list[dict[str, Any]] = []
        self.current_entrypoint: dict[str, Any] = {}

    def _clean(self):
        self.collected_entrypoints = []
        self.current_entrypoint = {}

    def generate_response(self, chat: Chat, user_message: HumanMessage) -> TrackResponse:
        sid = chat.context.session_id
        if sid != self.last_session_id:
            self.last_session_id = sid
            self._clean()

        state = chat.get_last_state(default=S.EMPTY)
        text = user_message.text.strip()
        logger.info(f"Processing request, state='{state}', text='{text}'")
        # data = (user_message.command or {}).get("query", "")

        # Get wizard data from last AI message
        wizard_data = {}
        last_ai = next((m for m in reversed(chat.messages) if isinstance(m, AIMessage)), None)
        if last_ai and last_ai.extra:
            wizard_data = last_ai.extra.get("wizard_data", {})
            self.collected_entrypoints = wizard_data.get("collected_entrypoints", [])
            self.current_entrypoint = wizard_data.get("current_entrypoint", {})

        if state == S.EMPTY:
            return S.CHOOSE_PROVIDER, self._create_provider_selection()

        elif state == S.CHOOSE_PROVIDER:
            if text == "Gigachat":
                self.current_entrypoint = {"provider": "gigachat"}
                return S.GIGACHAT_BASE_URL, self._ask_gigachat_base_url()
            elif text == "OpenRouter":
                self.current_entrypoint = {"provider": "openrouter"}
                return S.OPENROUTER_MODEL_ID, self._ask_openrouter_model_id()
            elif text.lower() == "exit":
                if not self.collected_entrypoints:
                    config = EntrypointsConfig(entrypoints={}, default_entrypoint_key='', warmup=False)
                    return S.FINAL, config.model_dump_json(indent=2)
                if len(self.collected_entrypoints) == 1:
                    default_key = self.collected_entrypoints[0]['key']
                    entrypoints_dict = {ep["key"]: EntrypointConfig(**ep) for ep in self.collected_entrypoints}
                    config = EntrypointsConfig(
                        entrypoints=entrypoints_dict,
                        default_entrypoint_key=default_key,
                        warmup=False,
                    )
                    return S.FINAL, config.model_dump_json(indent=2)
                return S.FINAL_DEFAULT_ENTRYPOINT, self._ask_default_entrypoint()
            else:
                return S.CHOOSE_PROVIDER, self._create_provider_selection("Select again")
                

        # Gigachat flow
        elif state == S.GIGACHAT_BASE_URL:
            base_url = text.strip().strip(".")
            
            if base_url and not base_url.startswith("http"):
                return S.GIGACHAT_BASE_URL, f"Unexpected base_url: {base_url}, try again"
            
            self.current_entrypoint["base_url"] = base_url or "https://gigachat.devices.sberbank.ru/api/v1"
            return S.GIGACHAT_AUTH_METHOD, self._ask_gigachat_auth_method()

        elif state == S.GIGACHAT_AUTH_METHOD:
            self.current_entrypoint["auth_method"] = text
            if text == "user/password":
                return S.GIGACHAT_USER, self._ask_gigachat_user()
            else:
                return S.GIGACHAT_CLIENT_ID, self._ask_gigachat_client_id()

        elif state == S.GIGACHAT_USER:
            self.current_entrypoint["user"] = text
            return S.GIGACHAT_PASSWORD, self._ask_gigachat_password()

        elif state == S.GIGACHAT_PASSWORD:
            self.current_entrypoint["password"] = text
            return S.GIGACHAT_CAPTION, self._ask_gigachat_caption()

        elif state == S.GIGACHAT_CLIENT_ID:
            self.current_entrypoint["client_id"] = text
            return S.GIGACHAT_CLIENT_SECRET, self._ask_gigachat_client_secret()

        elif state == S.GIGACHAT_CLIENT_SECRET:
            self.current_entrypoint["client_secret"] = text
            return S.GIGACHAT_CAPTION, self._ask_gigachat_caption()

        elif state == S.GIGACHAT_CAPTION:
            self.current_entrypoint["caption"] = text or "Gigachat"
            return S.GIGACHAT_KEY, make_content(text=f"Enter entrypoint key (default: {GIGACHAT_DEFAULT_KEY}):", widget=Widget.make_buttons(GIGACHAT_KEYS))

        elif state == S.GIGACHAT_KEY:
            text = text.strip().strip('..')
            if text and text not in GIGACHAT_KEYS:
                return S.GIGACHAT_KEY, make_content(text=f"Invalid gigachat key: {text}, choose again", widget=Widget.make_buttons(GIGACHAT_KEYS))

            key = text or GIGACHAT_DEFAULT_KEY
            self.current_entrypoint["key"] = key
            self._finalize_gigachat_entrypoint()
            return S.CHOOSE_PROVIDER, self._create_provider_selection()

        elif state == S.OPENROUTER_MODEL_ID:
            if not text:
                return S.OPENROUTER_MODEL_ID, self._ask_openrouter_model_id()
            self.current_entrypoint["model_id"] = text
            return S.OPENROUTER_API_KEY, self._ask_openrouter_api_key()

        elif state == S.OPENROUTER_API_KEY:
            if not text:
                return S.OPENROUTER_API_KEY, self._ask_openrouter_api_key()
            self.current_entrypoint["api_key"] = text
            return S.OPENROUTER_CAPTION, self._ask_openrouter_caption()

        elif state == S.OPENROUTER_CAPTION:
            self.current_entrypoint["caption"] = text or self.current_entrypoint["model_id"]
            return S.OPENROUTER_KEY, self._ask_openrouter_key()

        elif state == S.OPENROUTER_KEY:
            self.current_entrypoint["key"] = text or f"open-router_{self.current_entrypoint['model_id']}"
            self._finalize_openrouter_entrypoint()
            return S.CHOOSE_PROVIDER, self._create_provider_selection()

        # Finalization
        elif state == S.FINAL_DEFAULT_ENTRYPOINT:
            default_key = text or self.collected_entrypoints[0]["key"]
            entrypoints_dict = {ep["key"]: EntrypointConfig(**ep) for ep in self.collected_entrypoints}
            config = EntrypointsConfig(
                entrypoints=entrypoints_dict,
                default_entrypoint_key=default_key,
                warmup=False,
            )
            return S.FINAL, config.model_dump_json(indent=2)

        elif state == S.FINAL:
            return S.FINAL, 'Exit'

        # Fallback
        return S.FINAL, "Invalid input. Exit"

    def _create_provider_selection(self, text=None) -> dict:
        text = text or f"Total configured entrypoints: {len(self.collected_entrypoints)}. What would you like to configure?"
        return make_content(
            text=text, widget=Widget.make_buttons(["Gigachat", "OpenRouter", "Exit"])
        )

    def _ask_gigachat_base_url(self) -> dict:
        gigachat_base_url = "https://gigachat.devices.sberbank.ru/api/v1"
        return make_content(
            text=f"Enter GigaChat BASE_URL (default: {gigachat_base_url}):",
            widget=Widget.make_buttons([gigachat_base_url]),
        )

    def _ask_gigachat_auth_method(self) -> dict:
        return make_content(
            text="Select authentication method ( default: client_id/client_secret )",
            widget=Widget.make_buttons(["user/password", "client_id/client_secret"]),
        )

    def _ask_gigachat_user(self) -> dict:
        return make_content(text="Enter username:")

    def _ask_gigachat_password(self) -> dict:
        return make_content(text="Enter password:")

    def _ask_gigachat_client_id(self) -> dict:
        return make_content(text="Enter client ID:")

    def _ask_gigachat_client_secret(self) -> dict:
        return make_content(text="Enter client secret:")

    def _ask_gigachat_caption(self) -> dict:
        return make_content(text="Enter display name (default: Gigachat):", widget=Widget.make_buttons(["Gigachat"]))

    def _ask_openrouter_model_id(self) -> dict:
        buttons = [
            "google/gemini-2.0-flash-001",
            "deepseek/deepseek-chat-v3-0324",
            "meta-llama/llama-3.1-8b-instruct",
            "openai/gpt-3.5-turbo",
        ]
        return make_content(text="Enter model ID", widget=Widget.make_buttons(buttons))

    def _ask_openrouter_api_key(self) -> dict:
        return make_content(text="Enter API key:")

    def _ask_openrouter_caption(self) -> dict:
        model_id = self.current_entrypoint.get("model_id", "")
        return make_content(text=f"Enter display name (default: {model_id}):", widget=Widget.make_buttons([model_id]))

    def _ask_openrouter_key(self) -> dict:
        model_id = self.current_entrypoint.get("model_id", "")
        default_key = f"open-router_{model_id}"
        return make_content(
            text=f"Enter entrypoint key (default: {default_key}):", widget=Widget.make_buttons([default_key])
        )

    def _ask_default_entrypoint(self) -> dict:
        keys = [ep["key"] for ep in self.collected_entrypoints]
        return make_content(text="Select default entrypoint:", widget=Widget.make_buttons(keys))

    # Data processing methods
    def _finalize_gigachat_entrypoint(self):
        entrypoint = {
            "key": self.current_entrypoint["key"],
            "name": "gigachat",
            "caption": self.current_entrypoint["caption"],
            "args": {
                "base_url": self.current_entrypoint["base_url"],
#                "auth_method": self.current_entrypoint["auth_method"],
            },
        }

        if self.current_entrypoint["auth_method"] == "user/password":
            entrypoint["args"]["user"] = self.current_entrypoint["user"]
            entrypoint["args"]["password"] = self.current_entrypoint["password"]
        else:
            entrypoint["args"]["client_id"] = self.current_entrypoint["client_id"]
            entrypoint["args"]["client_secret"] = self.current_entrypoint["client_secret"]

        self.collected_entrypoints.append(entrypoint)
        self.current_entrypoint = {}

    def _finalize_openrouter_entrypoint(self):
        entrypoint = {
            "key": self.current_entrypoint["key"],
            "name": "open-router",
            "caption": self.current_entrypoint["caption"],
            "args": {
                "model_id": self.current_entrypoint["model_id"],
                "api_key": self.current_entrypoint["api_key"],
                "on_error": "fail",
            },
        }
        self.collected_entrypoints.append(entrypoint)
        self.current_entrypoint = {}

    def _save_wizard_data(self) -> dict[str, Any]:
        return {
            "collected_entrypoints": self.collected_entrypoints,
            "current_entrypoint": self.current_entrypoint,
        }

    def _response_with_data(self, state: str, content: dict) -> tuple[str, dict]:
        return state, content
