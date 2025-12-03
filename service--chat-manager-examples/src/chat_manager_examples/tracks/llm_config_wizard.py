from enum import StrEnum
from typing import Any

from loguru import logger
from mmar_llm import LLMConfig, LLMEndpointConfig
from mmar_mapi import AIMessage, Chat, HumanMessage, make_content
from mmar_mapi.models.widget import Widget
from mmar_mapi.tracks import SimpleTrack, TrackResponse

from chat_manager_examples.config import DOMAINS

STATES = [
    "EMPTY",  # Initial state
    "CHOOSE_PROVIDER",  # Choosing provider type
    "GIGACHAT_BASE_URL",  # Entering GigaChat BASE_URL
    "GIGACHAT_AUTH_METHOD",  # Choosing auth method
    "GIGACHAT_USER",  # Entering username
    "GIGACHAT_PASSWORD",  # Entering password
    "GIGACHAT_CLIENT_ID",  # Entering client ID
    "GIGACHAT_CLIENT_SECRET",  # Entering client secret
    "GIGACHAT_AUTHORIZATION_KEY",  # Entering authorization_key
    "GIGACHAT_CAPTION",  # Entering caption
    "GIGACHAT_KEY",  # Entering key
    "OPENROUTER_MODEL_ID",  # Entering OpenRouter model ID
    "OPENROUTER_API_KEY",  # Entering API key
    "OPENROUTER_CAPTION",  # Entering caption
    "OPENROUTER_KEY",  # Entering key
    "FINAL_DEFAULT_ENDPOINT",  # Setting default endpoint
    "FINAL",
]
S = StrEnum("State", STATES)

GIGACHAT_DEFAULT_KEY = "giga-max-2"


class LLMConfigWizard(SimpleTrack):
    DOMAIN = DOMAINS.examples
    CAPTION = "🧙‍♂️🤖 LLM Config Wizard"

    def __init__(self):
        # todo eliminate hotfixes: track should be stateless
        self.last_session_id = None
        self.collected_endpoints: list[dict[str, Any]] = []
        self.current_endpoint: dict[str, Any] = {}

    def _clean(self):
        self.collected_endpoints = []
        self.current_endpoint = {}

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
            self.collected_endpoints = wizard_data.get("collected_endpoints", [])
            self.current_endpoint = wizard_data.get("current_endpoint", {})

        if state == S.EMPTY:
            return S.CHOOSE_PROVIDER, self._create_provider_selection()

        elif state == S.CHOOSE_PROVIDER:
            if text == "Gigachat":
                self.current_endpoint = {"provider": "gigachat"}
                return S.GIGACHAT_BASE_URL, self._ask_gigachat_base_url()
            elif text == "OpenRouter":
                self.current_endpoint = {"provider": "openrouter"}
                return S.OPENROUTER_MODEL_ID, self._ask_openrouter_model_id()
            elif text.lower() == "exit":
                if not self.collected_endpoints:
                    config = LLMConfig(endpoints=[], default_endpoint_key="", warmup=False)
                    return S.FINAL, config.model_dump_json(indent=2)
                if len(self.collected_endpoints) == 1:
                    default_key = self.collected_endpoints[0]["key"]
                    config = LLMConfig(
                        endpoints=self.collected_endpoints,
                        default_endpoint_key=default_key,
                        warmup=False,
                    )
                    return S.FINAL, config.model_dump_json(indent=2)
                return S.FINAL_DEFAULT_ENDPOINT, self._ask_default_endpoint()
            else:
                return S.CHOOSE_PROVIDER, self._create_provider_selection("Select again")

        # Gigachat flow
        elif state == S.GIGACHAT_BASE_URL:
            base_url = text.strip().strip(".")

            if base_url and not base_url.startswith("http"):
                return S.GIGACHAT_BASE_URL, f"Unexpected base_url: {base_url}, try again"

            self.current_endpoint["base_url"] = base_url or "https://gigachat.devices.sberbank.ru/api/v1"
            return S.GIGACHAT_AUTH_METHOD, self._ask_gigachat_auth_method()

        elif state == S.GIGACHAT_AUTH_METHOD:
            if text == "user/password":
                self.current_endpoint["auth_method"] = "user/password"
                return S.GIGACHAT_USER, self._ask_gigachat_user()
            elif text == "authorization_key":
                self.current_endpoint["auth_method"] = "authorization_key"
                return S.GIGACHAT_AUTHORIZATION_KEY, self._ask_gigachat_authorization_key()
            else:
                self.current_endpoint["auth_method"] = "client_id/client_secret"
                return S.GIGACHAT_CLIENT_ID, self._ask_gigachat_client_id()

        elif state == S.GIGACHAT_USER:
            self.current_endpoint["user"] = text
            return S.GIGACHAT_PASSWORD, self._ask_gigachat_password()

        elif state == S.GIGACHAT_PASSWORD:
            self.current_endpoint["password"] = text
            return S.GIGACHAT_CAPTION, self._ask_gigachat_caption()

        elif state == S.GIGACHAT_CLIENT_ID:
            self.current_endpoint["client_id"] = text
            return S.GIGACHAT_CLIENT_SECRET, self._ask_gigachat_client_secret()

        elif state == S.GIGACHAT_CLIENT_SECRET:
            self.current_endpoint["client_secret"] = text
            return S.GIGACHAT_CAPTION, self._ask_gigachat_caption()

        elif state == S.GIGACHAT_AUTHORIZATION_KEY:
            self.current_endpoint["authorization_key"] = text
            return S.GIGACHAT_CAPTION, self._ask_gigachat_caption()

        elif state == S.GIGACHAT_CAPTION:
            self.current_endpoint["caption"] = text or "Gigachat"
            return S.GIGACHAT_KEY, make_content(text=f"Enter endpoint key (default: {GIGACHAT_DEFAULT_KEY}):")

        elif state == S.GIGACHAT_KEY:
            if text == "":
                text = GIGACHAT_DEFAULT_KEY
            text = text.strip().strip("..")
            if not text:
                return S.GIGACHAT_KEY, make_content(text=f"Invalid gigachat key: '{text}', choose again")

            key = text or GIGACHAT_DEFAULT_KEY
            self.current_endpoint["key"] = key
            self._finalize_gigachat_endpoint()
            return S.CHOOSE_PROVIDER, self._create_provider_selection()

        elif state == S.OPENROUTER_MODEL_ID:
            if not text:
                return S.OPENROUTER_MODEL_ID, self._ask_openrouter_model_id()
            self.current_endpoint["model_id"] = text
            return S.OPENROUTER_API_KEY, self._ask_openrouter_api_key()

        elif state == S.OPENROUTER_API_KEY:
            if not text:
                return S.OPENROUTER_API_KEY, self._ask_openrouter_api_key()
            self.current_endpoint["api_key"] = text
            return S.OPENROUTER_CAPTION, self._ask_openrouter_caption()

        elif state == S.OPENROUTER_CAPTION:
            self.current_endpoint["caption"] = text or self.current_endpoint["model_id"]
            return S.OPENROUTER_KEY, self._ask_openrouter_key()

        elif state == S.OPENROUTER_KEY:
            self.current_endpoint["key"] = text or f"open-router_{self.current_endpoint['model_id']}"
            self._finalize_openrouter_endpoint()
            return S.CHOOSE_PROVIDER, self._create_provider_selection()

        # Finalization
        elif state == S.FINAL_DEFAULT_ENDPOINT:
            default_key = text or self.collected_endpoints[0]["key"]
            endpoints_dict = {ep["key"]: LLMEndpointConfig(**ep) for ep in self.collected_endpoints}
            config = LLMConfig(
                endpoints=endpoints_dict,
                default_endpoint_key=default_key,
                warmup=False,
            )
            return S.FINAL, config.model_dump_json(indent=2)

        elif state == S.FINAL:
            return S.FINAL, "Exit"

        # Fallback
        return S.FINAL, "Invalid input. Exit"

    def _create_provider_selection(self, text=None) -> dict:
        text = text or f"Total configured endpoints: {len(self.collected_endpoints)}. What would you like to configure?"
        return make_content(text=text, widget=Widget.make_buttons(["Gigachat", "OpenRouter", "Exit"]))

    def _ask_gigachat_base_url(self) -> dict:
        gigachat_base_url = "https://gigachat.devices.sberbank.ru/api/v1"
        gigachat_alternative_url = "https://gigachat.sberdevices.ru/v1"
        return make_content(
            text=f"Enter GigaChat BASE_URL (default: {gigachat_base_url}):",
            widget=Widget.make_buttons([gigachat_base_url, gigachat_alternative_url]),
        )

    def _ask_gigachat_auth_method(self) -> dict:
        return make_content(
            text="Select authentication method ( default: client_id/client_secret )",
            widget=Widget.make_buttons(["user/password", "client_id/client_secret", "authorization_key"]),
        )

    def _ask_gigachat_user(self) -> dict:
        return make_content(text="Enter username:")

    def _ask_gigachat_password(self) -> dict:
        return make_content(text="Enter password:")

    def _ask_gigachat_client_id(self) -> dict:
        return make_content(text="Enter client ID:")

    def _ask_gigachat_authorization_key(self) -> dict:
        return make_content(text="Enter authorization_key:")

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
        model_id = self.current_endpoint.get("model_id", "")
        return make_content(text=f"Enter display name (default: {model_id}):", widget=Widget.make_buttons([model_id]))

    def _ask_openrouter_key(self) -> dict:
        model_id = self.current_endpoint.get("model_id", "")
        default_key = f"open-router_{model_id}"
        return make_content(
            text=f"Enter endpoint key (default: {default_key}):", widget=Widget.make_buttons([default_key])
        )

    def _ask_default_endpoint(self) -> dict:
        keys = [ep["key"] for ep in self.collected_endpoints]
        return make_content(text="Select default endpoint:", widget=Widget.make_buttons(keys))

    # Data processing methods
    def _finalize_gigachat_endpoint(self):
        endpoint = {
            "key": self.current_endpoint["key"],
            "descriptor": "gigachat",
            "caption": self.current_endpoint["caption"],
            "args": {
                "base_url": self.current_endpoint["base_url"],
                #                "auth_method": self.current_endpoint["auth_method"],
            },
        }

        auth_method = self.current_endpoint["auth_method"]
        if auth_method == "user/password":
            endpoint["args"]["user"] = self.current_endpoint["user"]
            endpoint["args"]["password"] = self.current_endpoint["password"]
        elif auth_method == "client_id/client_secret":
            endpoint["args"]["client_id"] = self.current_endpoint["client_id"]
            endpoint["args"]["client_secret"] = self.current_endpoint["client_secret"]
        elif auth_method == "authorization_key":
            endpoint["args"]["authorization_key"] = self.current_endpoint["authorization_key"]
        else:
            logger.error(f"Unexpected auth_method: {auth_method}")

        self.collected_endpoints.append(endpoint)
        self.current_endpoint = {}

    def _finalize_openrouter_endpoint(self):
        endpoint = {
            "key": self.current_endpoint["key"],
            "descriptor": "openrouter",
            "caption": self.current_endpoint["caption"],
            "args": {
                "model_id": self.current_endpoint["model_id"],
                "api_key": self.current_endpoint["api_key"],
            },
        }
        self.collected_endpoints.append(endpoint)
        self.current_endpoint = {}

    def _save_wizard_data(self) -> dict[str, Any]:
        return {
            "collected_endpoints": self.collected_endpoints,
            "current_endpoint": self.current_endpoint,
        }

    def _response_with_data(self, state: str, content: dict) -> tuple[str, dict]:
        return state, content
