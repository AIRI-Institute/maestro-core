from enum import StrEnum
from typing import Any, cast

from loguru import logger
from mmar_mapi import AIMessage, Chat, HumanMessage, make_content
from mmar_mapi.models.widget import Widget
from mmar_mapi.tracks import SimpleTrack, TrackResponse

from chat_manager_examples.config import DOMAINS

STATES = [  # type: ignore[misc]
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
    "GIGACHAT_MODEL_ID",  # Entering model ID
    "OPENAI_MODEL_ID",  # Entering OpenAI model ID
    "OPENAI_API_KEY",  # Entering API key
    "OPENAI_CAPTION",  # Entering caption
    "OPENAI_MODEL_ID_FINAL",  # Entering final model ID
    "FINAL_DEFAULT_MODEL",  # Setting default model
    "FINAL",
]
S = cast(StrEnum, StrEnum("State", STATES))

GIGACHAT_DEFAULT_KEY = "giga-max-2"


class LLMConfigWizard(SimpleTrack):
    DOMAIN = DOMAINS.examples
    CAPTION = "🧙‍♂️🤖 LLM Config Wizard"

    def __init__(self) -> None:
        # todo eliminate hotfixes: track should be stateless
        self.last_session_id: str | None = None
        self.collected_connections: list[dict[str, Any]] = []
        self.current_connection: dict[str, Any] = {}

    def _clean(self):
        self.collected_connections = []
        self.current_connection = {}

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
            self.collected_connections = wizard_data.get("collected_connections", [])
            self.current_connection = wizard_data.get("current_connection", {})

        if state == S.EMPTY:
            return S.CHOOSE_PROVIDER, self._create_provider_selection()

        elif state == S.CHOOSE_PROVIDER:
            if text == "Gigachat":
                self.current_connection = {"provider": "gigachat"}
                return S.GIGACHAT_BASE_URL, self._ask_gigachat_base_url()
            elif text == "OpenAI":
                self.current_connection = {"provider": "openai"}
                return S.OPENAI_MODEL_ID, self._ask_openai_model_id()
            elif text.lower() == "exit":
                if not self.collected_connections:
                    return S.FINAL, self._generate_toml_config()
                if len(self.collected_connections) == 1:
                    default_key = self.collected_connections[0]["key"]
                    return S.FINAL, self._generate_toml_config(default_key)
                return S.FINAL_DEFAULT_MODEL, self._ask_default_model()
            else:
                return S.CHOOSE_PROVIDER, self._create_provider_selection("Select again")

        # Gigachat flow
        elif state == S.GIGACHAT_BASE_URL:
            base_url = text.strip().strip(".")

            if base_url and not base_url.startswith("http"):
                return S.GIGACHAT_BASE_URL, f"Unexpected base_url: {base_url}, try again"

            self.current_connection["base_url"] = base_url or "https://gigachat.devices.sberbank.ru/api/v1"
            return S.GIGACHAT_AUTH_METHOD, self._ask_gigachat_auth_method()

        elif state == S.GIGACHAT_AUTH_METHOD:
            if text == "user/password":
                self.current_connection["auth_method"] = "user/password"
                return S.GIGACHAT_USER, self._ask_gigachat_user()
            elif text == "authorization_key":
                self.current_connection["auth_method"] = "authorization_key"
                return S.GIGACHAT_AUTHORIZATION_KEY, self._ask_gigachat_authorization_key()
            else:
                self.current_connection["auth_method"] = "client_id/client_secret"
                return S.GIGACHAT_CLIENT_ID, self._ask_gigachat_client_id()

        elif state == S.GIGACHAT_USER:
            self.current_connection["user"] = text
            return S.GIGACHAT_PASSWORD, self._ask_gigachat_password()

        elif state == S.GIGACHAT_PASSWORD:
            self.current_connection["password"] = text
            return S.GIGACHAT_CAPTION, self._ask_gigachat_caption()

        elif state == S.GIGACHAT_CLIENT_ID:
            self.current_connection["client_id"] = text
            return S.GIGACHAT_CLIENT_SECRET, self._ask_gigachat_client_secret()

        elif state == S.GIGACHAT_CLIENT_SECRET:
            self.current_connection["client_secret"] = text
            return S.GIGACHAT_CAPTION, self._ask_gigachat_caption()

        elif state == S.GIGACHAT_AUTHORIZATION_KEY:
            self.current_connection["authorization_key"] = text
            return S.GIGACHAT_CAPTION, self._ask_gigachat_caption()

        elif state == S.GIGACHAT_CAPTION:
            self.current_connection["caption"] = text or "Gigachat"
            return S.GIGACHAT_MODEL_ID, make_content(text=f"Enter model ID (default: {GIGACHAT_DEFAULT_KEY}):")

        elif state == S.GIGACHAT_MODEL_ID:
            if text == "":
                text = GIGACHAT_DEFAULT_KEY
            text = text.strip().strip("..")
            if not text:
                return S.GIGACHAT_MODEL_ID, make_content(text=f"Invalid model ID: '{text}', choose again")

            key = text or GIGACHAT_DEFAULT_KEY
            self.current_connection["key"] = key
            self._finalize_gigachat_connection()
            return S.CHOOSE_PROVIDER, self._create_provider_selection()

        elif state == S.OPENAI_MODEL_ID:
            if not text:
                return S.OPENAI_MODEL_ID, self._ask_openai_model_id()
            self.current_connection["model_id"] = text
            return S.OPENAI_API_KEY, self._ask_openai_api_key()

        elif state == S.OPENAI_API_KEY:
            if not text:
                return S.OPENAI_API_KEY, self._ask_openai_api_key()
            self.current_connection["api_key"] = text
            return S.OPENAI_CAPTION, self._ask_openai_caption()

        elif state == S.OPENAI_CAPTION:
            self.current_connection["caption"] = text or self.current_connection["model_id"]
            return S.OPENAI_MODEL_ID_FINAL, self._ask_openai_model_id_final()

        elif state == S.OPENAI_MODEL_ID_FINAL:
            self.current_connection["key"] = text or f"openai_{self.current_connection['model_id']}"
            self._finalize_openai_connection()
            return S.CHOOSE_PROVIDER, self._create_provider_selection()

        # Finalization
        elif state == S.FINAL_DEFAULT_MODEL:
            default_key = text or self.collected_connections[0]["key"]
            return S.FINAL, self._generate_toml_config(default_key)

        elif state == S.FINAL:
            return S.FINAL, "Exit"

        # Fallback
        return S.FINAL, "Invalid input. Exit"  # type: ignore[attr-defined]

    def _create_provider_selection(self, text=None) -> TrackResponse:
        if not text:
            text = f"Total configured connections: {len(self.collected_connections)}. What would you like to configure?"
        return make_content(text=text, widget=Widget.make_buttons(["Gigachat", "OpenAI", "Exit"]))

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

    def _ask_openai_model_id(self) -> dict:
        buttons = [
            "google/gemini-2.0-flash-001",
            "deepseek/deepseek-chat-v3-0324",
            "meta-llama/llama-3.1-8b-instruct",
            "openai/gpt-3.5-turbo",
        ]
        return make_content(text="Enter model ID", widget=Widget.make_buttons(buttons))

    def _ask_openai_api_key(self) -> dict:
        return make_content(text="Enter API key:")

    def _ask_openai_caption(self) -> dict:
        model_id = self.current_connection.get("model_id", "")
        return make_content(text=f"Enter display name (default: {model_id}):", widget=Widget.make_buttons([model_id]))

    def _ask_openai_model_id_final(self) -> dict:
        model_id = self.current_connection.get("model_id", "")
        default_key = f"openai_{model_id}"
        return make_content(text=f"Enter model ID (default: {default_key}):", widget=Widget.make_buttons([default_key]))

    def _ask_default_model(self) -> dict:
        keys = [ep["key"] for ep in self.collected_connections]
        return make_content(text="Select default model:", widget=Widget.make_buttons(keys))

    # Data processing methods
    def _finalize_gigachat_connection(self):
        auth_method = self.current_connection["auth_method"]
        connection = {
            "key": self.current_connection["key"],
            "provider": "gigachat",
            "connection_name": f"gigachat_{self.current_connection['key']}",
            "caption": self.current_connection["caption"],
            "base_url": self.current_connection["base_url"],
            "auth_method": auth_method,
        }

        if auth_method == "user/password":
            connection["user"] = self.current_connection["user"]
            connection["password"] = self.current_connection["password"]
        elif auth_method == "client_id/client_secret":
            connection["client_id"] = self.current_connection["client_id"]
            connection["client_secret"] = self.current_connection["client_secret"]
        elif auth_method == "authorization_key":
            connection["authorization_key"] = self.current_connection["authorization_key"]
        else:
            logger.error(f"Unexpected auth_method: {auth_method}")

        self.collected_connections.append(connection)
        self.current_connection = {}

    def _finalize_openai_connection(self):
        connection = {
            "key": self.current_connection["key"],
            "provider": "openai",
            "connection_name": f"openai-api_{self.current_connection['key']}",
            "caption": self.current_connection["caption"],
            "model_id": self.current_connection["model_id"],
            "api_key": self.current_connection["api_key"],
        }
        self.collected_connections.append(connection)
        self.current_connection = {}

    def _save_wizard_data(self) -> dict[str, Any]:
        return {
            "collected_connections": self.collected_connections,
            "current_connection": self.current_connection,
        }

    def _generate_toml_config(self, default_key: str | None = None) -> str:
        lines = ["# LLM Hub Configuration", ""]

        # Generate connections section
        for ep in self.collected_connections:
            conn_name = ep["connection_name"]
            lines.append(f"[connections.{conn_name}]")

            if ep["provider"] == "gigachat":
                lines.append('api_type = "gigachat"')
                lines.append(f'api_base = "{ep["base_url"]}"')
                auth_method = ep["auth_method"]
                if auth_method == "user/password":
                    lines.append(f'user = "{ep["user"]}"')
                    lines.append(f'password = "{ep["password"]}"')
                elif auth_method == "client_id/client_secret":
                    lines.append(f'client_id = "{ep["client_id"]}"')
                    lines.append(f'client_secret = "{ep["client_secret"]}"')
                elif auth_method == "authorization_key":
                    lines.append(f'authorization_key = "{ep["authorization_key"]}"')
            elif ep["provider"] == "openai":
                lines.append('api_type = "openai"')
                lines.append('api_base = "https://api.openai.com/v1"')
                lines.append(f'api_key = "{ep["api_key"]}"')
            lines.append("")

        # Generate routing section
        lines.append("[routing]")
        for ep in self.collected_connections:
            key = ep["key"]
            conn_name = ep["connection_name"]
            if ep["provider"] == "gigachat":
                lines.append(f'"{key}" = "{conn_name}.GigaChat"')
            elif ep["provider"] == "openai":
                model_id = ep["model_id"]
                lines.append(f'"{key}" = "{conn_name}.{model_id}"')
        lines.append("")

        # Generate model_info section
        for ep in self.collected_connections:
            key = ep["key"]
            lines.append(f"[model_info.{key}]")
            lines.append(f'caption = "{ep["caption"]}"')
            if default_key and key == default_key:
                lines.append("default = true")
            lines.append("")

        # Generate groups section
        lines.append("[groups.default]")
        lines.append("# Allow all connections")
        lines.append('allowed_connections = "*"')

        return "\n".join(lines)

    def _response_with_data(self, state: str, content: dict) -> tuple[str, dict]:
        return state, content
