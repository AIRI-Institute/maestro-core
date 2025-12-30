"""
Slot-filling track for conversational data collection.

TODO: Move to mmar-mimpl or mmar-mapi library.
"""

import copy
import re
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated, Any, Literal, Union, cast, get_args, get_origin

import pydantic
from pydantic import BaseModel, Field

from mmar_mapi import AIMessage, Chat, HumanMessage
from mmar_mapi.tracks import SimpleTrack, TrackResponse

# Special state for confirmation step
CONFIRMATION_STATE = "__CONFIRMATION__"


# =============================================================================
# Common text field patterns
# =============================================================================

TEXT_FIELD_PATTERNS = SimpleNamespace(
    email=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    phone=r"^[\d\s\+\-\(\)]+$",
)


# =============================================================================
# Module-level validation and parsing functions
# =============================================================================


def validate_text(
    value: str, min_length: int = 0, max_length: int | None = None, pattern: str | None = None
) -> tuple[bool, str | None]:
    """Validate text input with optional length and pattern validation."""
    if len(value) < min_length:
        return False, f"Must be at least {min_length} characters"
    if max_length and len(value) > max_length:
        return False, f"Must be at most {max_length} characters"
    if pattern and not re.match(pattern, value):
        return False, f"Must match pattern: {pattern}"
    return True, None


def validate_int(value: str, min_value: int | None = None, max_value: int | None = None) -> tuple[bool, str | None]:
    """Validate integer input with optional min/max validation."""
    try:
        int_value = int(value)
    except ValueError:
        return False, "Please enter a valid number"
    if min_value is not None and int_value < min_value:
        return False, f"Must be at least {min_value}"
    if max_value is not None and int_value > max_value:
        return False, f"Must be at most {max_value}"
    return True, None


def parse_int(value: str) -> int:
    """Parse string to integer."""
    return int(value)


def validate_float(
    value: str, min_value: float | None = None, max_value: float | None = None
) -> tuple[bool, str | None]:
    """Validate float input with optional min/max validation."""
    try:
        float_value = float(value)
    except ValueError:
        return False, "Please enter a valid number"
    if min_value is not None and float_value < min_value:
        return False, f"Must be at least {min_value}"
    if max_value is not None and float_value > max_value:
        return False, f"Must be at most {max_value}"
    return True, None


def parse_float(value: str) -> float:
    """Parse string to float."""
    return float(value)


def validate_choice(value: str, choices: list[str] | None, case_sensitive: bool = False) -> tuple[bool, str | None]:
    """Validate single choice from a list of options."""
    if choices is None:
        return True, None
    choices_str = ", ".join(choices)
    if case_sensitive:
        if value not in choices:
            return False, f"Must be one of: {choices_str}"
    else:
        choices_lower = {c.lower(): c for c in choices}
        if value.lower() not in choices_lower:
            return False, f"Must be one of: {choices_str}"
    return True, None


def parse_choice(value: str, choices: list[str] | None, case_sensitive: bool = False) -> str:
    """Parse choice value, handling case insensitivity."""
    if case_sensitive or not choices:
        return value
    choices_lower = {c.lower(): c for c in choices}
    return choices_lower.get(value.lower(), value)


def validate_bool(
    value: str,
    yes_values: set[str] | None = None,
    no_values: set[str] | None = None,
) -> tuple[bool, str | None]:
    """Validate boolean yes/no field."""
    yes_values = yes_values or {"yes", "y", "true", "1", "да", "д"}
    no_values = no_values or {"no", "n", "false", "0", "нет", "н"}
    lower_val = value.lower().strip()
    if lower_val in yes_values or lower_val in no_values:
        return True, None
    options = ", ".join(sorted(list(yes_values)[:3] + list(no_values)[:3]))
    return False, f"Please answer: {options}..."


def parse_bool(value: str, yes_values: set[str] | None = None) -> bool:
    """Parse boolean value from string."""
    yes_values = yes_values or {"yes", "y", "true", "1", "да", "д"}
    return value.lower().strip() in yes_values


def get_field_constraint(field_info, constraint_name: str) -> Any:
    """Extract a constraint value from field_info.metadata."""
    for metadata in field_info.metadata:
        if hasattr(metadata, constraint_name):
            value = getattr(metadata, constraint_name)
            if value is not None:
                return value
    return None


def get_field_type(field_info) -> str:
    """Get the field type ('str', 'int', 'float', 'bool', 'choice', 'email', 'phone', 'group')."""
    annotation = field_info.annotation

    # Handle Optional types
    if get_origin(annotation) is Union:
        args = get_args(annotation)
        # Get the first non-None type
        for arg in args:
            if arg is not type(None):
                annotation = arg
                break

    # Check for Literal (choices)
    if get_origin(annotation) is Literal:
        return "choice"

    # Check for nested BaseModel
    origin = get_origin(annotation)
    if origin is not None and origin is not list:
        # Check if it's a BaseModel subclass
        try:
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                return "group"
        except TypeError:
            pass

    # Basic types
    if annotation is str:
        # Check if it looks like an email field
        if "email" in str(field_info).lower():
            return "email"
        # Check if it looks like a phone field
        if "phone" in str(field_info).lower() or "tel" in str(field_info).lower():
            return "phone"
        return "str"
    elif annotation is int:
        return "int"
    elif annotation is float:
        return "float"
    elif annotation is bool:
        return "bool"

    return "str"  # Default to text field


# =============================================================================
# Module-level field helper functions
# =============================================================================


def is_field_required(required_fields: dict[str, bool], key: str) -> bool:
    """Check if a field is required based on the required fields dict.

    Handle compound keys for nested fields (e.g., "address.street").
    """
    if "." in key:
        parts = key.split(".")
        # For nested fields, check the parent group's requirement
        parent_key = parts[0]
        if parent_key in required_fields:
            return required_fields[parent_key]
        return True
    return required_fields.get(key, True)


def get_model_default(model: type[BaseModel], key: str) -> Any:
    """Get the default value from the Pydantic model for a field."""
    if key in model.model_fields:
        field_info = model.model_fields[key]
        if not field_info.is_required():
            return field_info.default
    return None


def find_field_index_by_key(flat_fields: list, key: str) -> int:
    """Find field index by key in flattened fields."""
    for i, field in enumerate(flat_fields):
        if field.key == key:
            return i
    raise ValueError(f"Field not found: {key}")


def get_first_uncollected_optional_field(
    flat_fields: list, required_fields: dict[str, bool], collected_values: dict[str, Any]
) -> str | None:
    """Get the first uncollected optional field, or None if all collected."""
    for field in flat_fields:
        if field.key not in collected_values and not is_field_required(required_fields, field.key):
            return field.key
    return None


class FieldDescriptor(BaseModel):
    """Overrides for default field behavior."""

    prompt: str | None = None
    description: str | None = None
    extract_key: str | None = None  # For document extraction
    skippable: bool | None = None
    choices: list[str] | None = None  # For ChoiceField

    # Validator is a callable, excluded from model serialization
    validator: Callable[[str], tuple[bool, str | None]] | None = Field(default=None, exclude=True)


class BaseFieldDescriptor(BaseModel):
    """Default settings applied to all fields unless overridden."""

    allow_skip: bool = True  # Allow skipping optional fields
    prompt_template: str = "Enter {description}:"

    # Maps target field key -> source field key
    # When source field is filled, use its value as default for target field
    default_strategy: dict[str, str] = Field(default_factory=dict)


class FormField(ABC):
    """Base class for form fields that can be filled via chat."""

    def __init__(
        self,
        *,
        key: str,
        prompt: str,
        description: str,
        skippable: bool = True,
        default: Any = None,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
        choices: list[str] | None = None,
    ):
        self.key = key
        self.prompt = prompt
        self.description = description
        self.skippable = skippable
        self.default = default
        self.validator = validator
        self.choices = choices

    def validate(self, value: str) -> tuple[bool, str | None]:
        """Returns (is_valid, error_message). Use validator callable if provided."""
        if self.validator is not None:
            return self.validator(value)
        return self._default_validate(value)

    def _default_validate(self, value: str) -> tuple[bool, str | None]:
        """Default validation when no validator callable is provided."""
        return True, None

    @abstractmethod
    def parse(self, value: str) -> Any:
        """Convert string to target type."""
        pass


class SlotFillingTrack(SimpleTrack):
    """Track for collecting structured data through form-like conversation.

    TODO: This is a legacy implementation. Should be moved to mmar-mimpl or mmar-mapi.
    """

    DOMAIN: str | None = None
    CAPTION: str | None = None

    def __init__(
        self,
        model: type[BaseModel],
        *,
        fields: list[FormField] | None = None,
        base_descriptor: BaseFieldDescriptor | None = None,
        field_descriptors: dict[str, FieldDescriptor] | None = None,
        exit_command: str = "exit",
        skip_command: str = "skip",
        confirm_at_end: bool = False,
    ):
        """
        Args:
            model: Pydantic BaseModel class defining the target structure
            fields: Optional list of FormFields. If not provided, will be auto-generated from model.
            base_descriptor: Default settings for all fields (prompt_template, default_strategy)
            field_descriptors: Optional overrides for prompts, validators, etc.
            exit_command: Command to exit the form early
            skip_command: Command to skip the current field
            confirm_at_end: Show summary and require confirmation
        """
        self.model = model
        self.base_descriptor = base_descriptor or BaseFieldDescriptor()
        self.field_descriptors = field_descriptors or {}
        self.exit_command = exit_command
        self.skip_command = skip_command
        self.confirm_at_end = confirm_at_end

        # Always auto-generate fields from model as base
        self._fields = self._create_fields_from_model()

        # If custom fields provided, merge them (override matching keys)
        if fields is not None:
            # Create a map of field key to custom field for quick lookup
            custom_fields_by_key = {f.key: f for f in fields}
            # Merge: use custom fields where provided, keep auto-generated for others
            merged_fields = []
            for auto_field in self._fields:
                if auto_field.key in custom_fields_by_key:
                    merged_fields.append(custom_fields_by_key[auto_field.key])
                else:
                    merged_fields.append(auto_field)
            # Add any custom fields that don't exist in model (for advanced use cases)
            for key, custom_field in custom_fields_by_key.items():
                if not any(f.key == key for f in self._fields):
                    merged_fields.append(custom_field)
            self._fields = merged_fields  # type: ignore[assignment]

        # Track which fields are required based on Pydantic model
        self._required_fields: dict[str, bool] = self._get_required_fields()

        self._field_to_group: dict[str, str | None] = {}  # field.key -> group.key or None
        self._field_to_list: dict[str, str | None] = {}  # field.key -> list.key or None
        self._list_fields: set[str] = set()  # Set of list field keys
        self._flat_fields: list[FormField] = self._flatten_fields()

    def _handle_skip_command_for_derive(self, field: FormField, collected_values: dict[str, Any]) -> str | None:
        """Handle skip command during session derivation. Return next field key."""
        # Use model default for optional fields, form field default otherwise
        if not is_field_required(self._required_fields, field.key):
            collected_values[field.key] = get_model_default(self.model, field.key)
        else:
            collected_values[field.key] = field.default
        # Move to next field (sequential only, don't jump to optional)
        idx = find_field_index_by_key(self._flat_fields, field.key)
        if idx + 1 < len(self._flat_fields):
            return self._flat_fields[idx + 1].key
        return None

    def _finalize_list_item(
        self, list_key: str, list_items: dict[str, list[dict[str, Any]]], current_list_item: dict[str, dict[str, Any]]
    ) -> None:
        """Finalize current list item and add to list_items."""
        if current_list_item.get(list_key):
            list_items[list_key].append(current_list_item[list_key])
            current_list_item[list_key] = {}

    def _handle_polymorphic_type_selection(
        self, list_key: str, selected_variant: str, parsed_value: Any, current_list_item: dict[str, dict[str, Any]]
    ) -> str | None:
        """Handle polymorphic list type selection - return next field key."""
        # After selecting type, jump to first field of the selected variant
        for f in self._flat_fields:
            if (
                self._field_to_list.get(f.key) == list_key
                and not f.key.endswith("._more")
                and self._field_to_group.get(f.key) == selected_variant
            ):
                # Set the auth_type in current_list_item
                if list_key not in current_list_item:
                    current_list_item[list_key] = {}
                current_list_item[list_key]["auth_type"] = parsed_value
                return f.key
        # No variant fields found - use next field
        idx = find_field_index_by_key(self._flat_fields, f"{list_key}._type")
        if idx + 1 < len(self._flat_fields):
            return self._flat_fields[idx + 1].key
        return None

    def _find_first_list_item_field(self, list_key: str) -> str | None:
        """Find first item field for a list (excluding _more fields)."""
        for f in self._flat_fields:
            if self._field_to_list.get(f.key) == list_key and not f.key.endswith("._more"):
                return f.key
        return None

    def _advance_to_next_field(self, field_key: str) -> str | None:
        """Get the next field key after the given field."""
        idx = find_field_index_by_key(self._flat_fields, field_key)
        if idx + 1 < len(self._flat_fields):
            return self._flat_fields[idx + 1].key
        return None

    def _process_field_input(
        self,
        field: FormField,
        user_text: str,
        collected_values: dict[str, Any],
        list_items: dict[str, list[dict[str, Any]]],
        current_list_item: dict[str, dict[str, Any]],
    ) -> str | None:
        """Process user input for a field - return next field key."""
        is_valid, _ = field.validate(user_text)
        if not is_valid:
            return field.key  # Stay on same field

        parsed_value = field.parse(user_text)
        list_key = self._field_to_list.get(field.key)

        if list_key and not field.key.endswith("._more"):
            # Extract field name from compound key (e.g., "name" from "fruits.name")
            field_name = field.key.split(".", 1)[1]
            # Special handling for polymorphic list _type field
            if field_name == "_type":
                return self._handle_polymorphic_type_selection(list_key, parsed_value, parsed_value, current_list_item)
            else:
                if list_key not in current_list_item:
                    current_list_item[list_key] = {}
                current_list_item[list_key][field_name] = parsed_value
                return None  # Stay in list processing
        elif field.key.endswith("._more"):
            # Handle "add another" prompt
            if list_key:
                self._finalize_list_item(list_key, list_items, current_list_item)
                if parsed_value.lower() == "yes":
                    return self._find_first_list_item_field(list_key)
                else:
                    return self._advance_to_next_field(field.key)
        else:
            # Regular field
            collected_values[field.key] = parsed_value
            return self._advance_to_next_field(field.key)
        return None

    def _skip_list_field(self, list_key: str, session: dict) -> None:
        """Skip a list field - move to field after _more."""
        more_key = f"{list_key}._more"
        next_key = self._advance_to_next_field(more_key)
        session["current_field_key"] = next_key

    def _skip_group_field(self, group_key: str, session: dict) -> None:
        """Skip a group field - skip all fields in the group."""
        group_field = next((f for f in self._fields if f.key == group_key), None)
        if group_field and isinstance(group_field, GroupField):
            for nested_field in group_field.fields:
                # Use model default for optional fields, form field default otherwise
                if not is_field_required(self._required_fields, nested_field.key):
                    session["collected_values"][nested_field.key] = get_model_default(self.model, nested_field.key)
                else:
                    session["collected_values"][nested_field.key] = nested_field.default
            # Move to field after the last field in the group
            last_nested_key = group_field.fields[-1].key
            next_key = self._get_next_field_key(last_nested_key)
            if next_key is None:
                next_key = self._get_first_uncollected_optional_field(session)
            session["current_field_key"] = next_key

    def _handle_list_field_value(
        self, current_key: str, parsed_value: Any, list_key: str, session: dict
    ) -> tuple[str | None, TrackResponse | None]:
        """Handle value for a list field. Returns (next_key, response) or (None, None) to continue normally."""
        # Extract field name from compound key (e.g., "name" from "fruits.name")
        field_name = current_key.split(".", 1)[1]

        # Special handling for polymorphic list _type field -> auth_type
        if field_name == "_type":
            field_name = "auth_type"
            # After selecting type, jump to first field of the selected variant
            selected_variant = parsed_value
            # Find first field of this variant
            next_key: str | None = None
            for f in self._flat_fields:
                if (
                    self._field_to_list.get(f.key) == list_key
                    and not f.key.endswith("._more")
                    and self._field_to_group.get(f.key) == selected_variant
                ):
                    next_key = f.key
                    break
            if next_key is None:
                next_key = self._get_next_field_key(current_key)
            if next_key is None:
                next_key = self._get_first_uncollected_optional_field(session)
            session["current_field_key"] = next_key
            return next_key, self._get_next_prompt_for_key(next_key, session)

        # Regular list field
        if "current_list_item" not in session:
            session["current_list_item"] = {}
        if list_key not in session["current_list_item"]:
            session["current_list_item"][list_key] = {}
        session["current_list_item"][list_key][field_name] = parsed_value
        return None, None  # Continue normal flow

    def _get_next_prompt_for_key(self, key: str | None, session: dict) -> TrackResponse:
        """Get prompt for a given field key."""
        if key is None:
            key = self._get_first_uncollected_optional_field(session)
        if key is None:
            # Form is complete - return a completion response
            return "final", "Form complete"  # type: ignore[return-value]
        field = next((f for f in self._flat_fields if f.key == key), None)
        if field is None:
            # Form is complete - return a completion response
            return "final", "Form complete"  # type: ignore[return-value]
        # Check for default strategy
        if field.key in self.base_descriptor.default_strategy:
            source_key = self.base_descriptor.default_strategy[field.key]
            if source_key in session["collected_values"]:
                default_value = session["collected_values"][source_key]
                prompt = f"{field.prompt} (default: {default_value})"
                return key, prompt  # type: ignore[return-value]
        return key, field.prompt  # type: ignore[return-value]

    def _handle_confirmation_response(self, text: str, chat: Chat, session: dict) -> TrackResponse | None:
        """Handle confirmation state response. Returns None if not in confirmation state."""
        current_key = session["current_field_key"]
        if current_key != CONFIRMATION_STATE:
            return None

        if text.lower() in ("yes", "y"):
            return self._handle_completion(chat, session)
        else:
            # User rejected - ask first question again
            session["confirmed"] = False
            session["current_field_key"] = self._flat_fields[0].key if self._flat_fields else None
            return self._get_next_prompt(chat, session)

    def _validate_and_parse_field(self, field: FormField, text: str) -> tuple[bool, str | None, Any | None]:
        """Validate and parse field input. Returns (is_valid, error_msg, parsed_value)."""
        is_valid, error_msg = field.validate(text)
        if not is_valid:
            return False, error_msg or "Invalid input", None

        parsed_value = field.parse(text)

        # Check custom validator if present
        if field.validator is not None:
            is_valid, error_msg = field.validator(parsed_value)
            if not is_valid:
                return False, error_msg or "Invalid input", None

        return True, None, parsed_value

    def _handle_skip_command(
        self, current_key: str, current_field: FormField, session: dict, chat: Chat
    ) -> TrackResponse | None:
        """Handle skip command. Returns None if not a skip command."""
        # Check if this field belongs to a list - skip entire list
        list_key = self._field_to_list.get(current_field.key)
        if list_key and not current_key.endswith("._more"):
            self._skip_list_field(list_key, session)
            return self._get_next_prompt(chat, session)

        # Check if this field belongs to a group - skip entire group
        group_key = self._field_to_group.get(current_field.key)
        if group_key:
            self._skip_group_field(group_key, session)
            return self._get_next_prompt(chat, session)

        # Use model default for optional fields, form field default otherwise
        if not is_field_required(self._required_fields, current_key):
            session["collected_values"][current_field.key] = get_model_default(self.model, current_key)
        else:
            session["collected_values"][current_field.key] = current_field.default
        next_key = self._get_next_field_key(current_key)
        if next_key is None:
            next_key = get_first_uncollected_optional_field(
                self._flat_fields, self._required_fields, session["collected_values"]
            )
        session["current_field_key"] = next_key
        return self._get_next_prompt(chat, session)

    def _handle_regular_field_storage(self, current_key: str, parsed_value: Any, session: dict) -> str | None:
        """Handle storing a regular field value. Returns next field key."""
        session["collected_values"][current_key] = parsed_value
        next_key = self._get_next_field_key(current_key)
        if next_key is None:
            next_key = self._get_first_uncollected_optional_field(session)
        session["current_field_key"] = next_key
        return next_key

    def _handle_more_field(
        self, list_key: str, parsed_value: str, current_key: str, session: dict, chat: Chat
    ) -> TrackResponse:
        """Handle the _more field for lists (add another item)."""
        if "current_list_item" not in session:
            session["current_list_item"] = {}
        if "list_items" not in session:
            session["list_items"] = {k: [] for k in self._list_fields}
        # Finalize current item if exists
        if session["current_list_item"].get(list_key):
            session["list_items"][list_key].append(session["current_list_item"][list_key])
            session["current_list_item"][list_key] = {}
        # Check if user wants more
        if parsed_value.lower() == "yes":
            first_item_field = self._find_first_list_item_field(list_key)
            session["current_field_key"] = first_item_field
            return self._get_next_prompt(chat, session)
        else:
            # Done with this list - move to next field after _more
            next_key = self._advance_to_next_field(current_key)
            session["current_field_key"] = next_key
            return self._get_next_prompt(chat, session)

    def _derive_session_from_history(self, chat: Chat) -> dict:
        """Derive session state by parsing message history.

        Returns dict with:
        - current_field_key: The field we're currently waiting for
        - collected_values: Values collected so far from completed fields
        - confirmed: Whether user has confirmed (for confirm_at_end)
        - list_items: Dict of {list_key: [completed_items]}
        - current_list_item: Dict of {list_key: {field: value}} for item being built
        """
        collected_values: dict[str, Any] = {}
        list_items: dict[str, list[dict[str, Any]]] = {k: [] for k in self._list_fields}
        current_list_item: dict[str, dict[str, Any]] = {}
        current_field_key = self._flat_fields[0].key if self._flat_fields else None
        confirmed = False

        # Parse message history to find collected values
        messages = chat.messages
        i = 0
        while i < len(messages):
            msg = messages[i]

            # Look for AI messages with field state
            if getattr(msg, "type", None) == "ai":
                ai_msg = cast(AIMessage, msg)
                if ai_msg.state:
                    if ai_msg.state == "final":
                        # Form is complete
                        current_field_key = None
                        break
                    elif ai_msg.state == CONFIRMATION_STATE:
                        current_field_key = CONFIRMATION_STATE
                        confirmed = True
                    elif ai_msg.state in self._field_to_group or ai_msg.state in self._field_to_list:
                        # This is a valid field state
                        # Check if the next user message provided a value for this field
                        if i + 1 < len(messages) and getattr(messages[i + 1], "type", None) == "human":
                            user_msg = cast(HumanMessage, messages[i + 1])
                            user_text = user_msg.text.strip()

                            # Only process if there's an AI response after the human message
                            # (meaning the user input was already responded to)
                            if i + 2 < len(messages) and getattr(messages[i + 2], "type", None) == "ai":
                                # Check if user exited or skipped
                                if user_text.lower() == self.exit_command.lower():
                                    # User exited - use collected values so far
                                    current_field_key = None
                                    break
                                elif user_text.lower() == self.skip_command.lower():
                                    # User skipped - set default value
                                    field = next((f for f in self._flat_fields if f.key == ai_msg.state), None)
                                    if field:
                                        next_key = self._handle_skip_command_for_derive(field, collected_values)
                                        current_field_key = next_key
                                    i += 2  # Skip AI prompt and human response
                                    continue
                                else:
                                    # Try to validate and parse the user input
                                    field = next((f for f in self._flat_fields if f.key == ai_msg.state), None)
                                    if field:
                                        next_key = self._process_field_input(
                                            field, user_text, collected_values, list_items, current_list_item
                                        )
                                        if next_key is not None:
                                            current_field_key = next_key
                                    i += 2  # Skip AI prompt and human response
                                    continue
                            else:
                                # No AI response after human message - this is the current pending input
                                current_field_key = ai_msg.state
                        else:
                            # No user response after AI - we're at this field
                            current_field_key = ai_msg.state
            i += 1

        return {
            "current_field_key": current_field_key,
            "collected_values": collected_values,
            "confirmed": confirmed,
            "list_items": list_items,
            "current_list_item": current_list_item,
        }

    def _get_required_fields(self) -> dict[str, bool]:
        """Get which fields are required from the Pydantic model."""
        required: dict[str, bool] = {}
        for field_name, field_info in self.model.model_fields.items():
            required[field_name] = field_info.is_required()
        return required

    def _create_fields_from_model(self) -> list[FormField]:
        """Introspect Pydantic model and create FormFields."""
        fields: list[FormField] = []

        for field_name, field_info in self.model.model_fields.items():
            descriptor = self.field_descriptors.get(field_name, FieldDescriptor())

            # Determine prompt
            prompt = descriptor.prompt or self._generate_prompt(field_name, field_info)

            # Determine description
            description = descriptor.description or field_info.description or field_name

            # Get field type and constraints
            field_type = get_field_type(field_info)
            default = field_info.default if not field_info.is_required() else None

            # Create appropriate FormField based on type
            field = self._create_field(
                key=field_name,
                prompt=prompt,
                description=description,
                field_type=field_type,
                field_info=field_info,
                default=default,
                descriptor=descriptor,
            )
            fields.append(field)

        return fields

    def _generate_prompt(self, field_name: str, field_info) -> str:
        """Generate a default prompt for a field."""
        description = field_info.description or field_name
        return self.base_descriptor.prompt_template.format(description=description)

    def _create_field(
        self,
        key: str,
        prompt: str,
        description: str,
        field_type: str,
        field_info,
        default: Any,
        descriptor: FieldDescriptor,
    ) -> FormField:
        """Create a FormField based on type and constraints."""
        is_required = field_info.is_required()

        if field_type == "group":
            # Nested BaseModel - create GroupField
            nested_fields = self._create_fields_from_model_nested(field_info.annotation, key, descriptor)
            return GroupField(
                key=key,
                prompt=prompt,
                description=description,
                fields=nested_fields,
                skippable=descriptor.skippable if descriptor.skippable is not None else not is_required,
            )

        elif field_type == "int":
            # Get min/max from Field constraints
            ge = get_field_constraint(field_info, "ge")
            le = get_field_constraint(field_info, "le")
            return IntField(
                key=key,
                prompt=prompt,
                description=description,
                min_value=ge,
                max_value=le,
                skippable=descriptor.skippable if descriptor.skippable is not None else not is_required,
                default=default if default is not None else 0,
                validator=descriptor.validator,
            )

        elif field_type == "float":
            ge = get_field_constraint(field_info, "ge")
            le = get_field_constraint(field_info, "le")
            return FloatField(
                key=key,
                prompt=prompt,
                description=description,
                min_value=ge,
                max_value=le,
                skippable=descriptor.skippable if descriptor.skippable is not None else not is_required,
                default=default if default is not None else 0.0,
                validator=descriptor.validator,
            )

        elif field_type == "choice":
            choices = list(get_args(field_info.annotation))
            # Filter out None if it's Optional
            choices = [c for c in choices if c is not None]
            return ChoiceField(
                key=key,
                prompt=prompt,
                description=description,
                choices=descriptor.choices or choices,
                skippable=descriptor.skippable if descriptor.skippable is not None else not is_required,
                default=default,
                validator=descriptor.validator,
            )

        elif field_type == "bool":
            return BoolField(
                key=key,
                prompt=prompt,
                description=description,
                skippable=descriptor.skippable if descriptor.skippable is not None else not is_required,
                default=default if default is not None else False,
                validator=descriptor.validator,
            )

        elif field_type == "email":
            return TextField(
                key=key,
                prompt=prompt,
                description=description,
                pattern=TEXT_FIELD_PATTERNS.email,
                skippable=descriptor.skippable if descriptor.skippable is not None else not is_required,
                default=default if default is not None else "",
                validator=descriptor.validator,
            )

        elif field_type == "phone":
            return TextField(
                key=key,
                prompt=prompt,
                description=description,
                pattern=TEXT_FIELD_PATTERNS.phone,
                skippable=descriptor.skippable if descriptor.skippable is not None else not is_required,
                default=default if default is not None else "",
                validator=descriptor.validator,
            )

        else:  # str or default
            # Get min/max length from constraints
            min_length = get_field_constraint(field_info, "min_length")
            max_length = get_field_constraint(field_info, "max_length")
            return TextField(
                key=key,
                prompt=prompt,
                description=description,
                min_length=min_length if min_length else 0,
                max_length=max_length,
                skippable=descriptor.skippable if descriptor.skippable is not None else not is_required,
                default=default if default is not None else "",
                validator=descriptor.validator,
            )

    def _create_fields_from_model_nested(
        self, model_class: type[BaseModel], parent_key: str, parent_descriptor: FieldDescriptor
    ) -> list[FormField]:
        """Create FormFields for nested BaseModel."""
        fields: list[FormField] = []

        for field_name, field_info in model_class.model_fields.items():
            # Build compound key for nested field
            compound_key = f"{parent_key}.{field_name}"
            descriptor = parent_descriptor  # TODO: support nested field_descriptors

            prompt = self._generate_prompt(field_name, field_info)
            description = field_info.description or field_name
            field_type = get_field_type(field_info)
            default = field_info.default if not field_info.is_required() else None

            field = self._create_field(
                key=compound_key,
                prompt=prompt,
                description=description,
                field_type=field_type,
                field_info=field_info,
                default=default,
                descriptor=descriptor,
            )
            fields.append(field)

        return fields

    def _flatten_fields(self) -> list[FormField]:
        """Flatten nested GroupFields and ListFields for sequential processing."""
        fields: list[FormField] = []
        for field in self._fields:
            if isinstance(field, GroupField):
                for nested_field in field.fields:
                    self._field_to_group[nested_field.key] = field.key
                    fields.append(nested_field)
            elif isinstance(field, ListField):
                self._list_fields.add(field.key)
                # Add item fields with compound keys like "fruits.name"
                for item_field in field.item_fields:
                    compound_key = f"{field.key}.{item_field.key}"
                    self._field_to_list[compound_key] = field.key
                    # Create a NEW instance with the compound key, don't modify the original!
                    new_field: FormField  # type: ignore[no-redef]
                    if isinstance(item_field, TextField):
                        new_field = TextField(
                            key=compound_key,
                            prompt=item_field.prompt,
                            description=item_field.description,
                            min_length=item_field.min_length,
                            max_length=item_field.max_length,
                            pattern=item_field.pattern,
                            skippable=item_field.skippable,
                            default=item_field.default,
                            validator=item_field.validator,
                        )
                    elif isinstance(item_field, IntField):
                        new_field = IntField(
                            key=compound_key,
                            prompt=item_field.prompt,
                            description=item_field.description,
                            min_value=item_field.min_value,
                            max_value=item_field.max_value,
                            skippable=item_field.skippable,
                            default=item_field.default,
                            validator=item_field.validator,
                        )
                    elif isinstance(item_field, ChoiceField):
                        new_field = ChoiceField(
                            key=compound_key,
                            prompt=item_field.prompt,
                            description=item_field.description,
                            choices=item_field.choices or [],
                            case_sensitive=getattr(item_field, "case_sensitive", False),
                            skippable=item_field.skippable,
                            default=item_field.default,
                            validator=item_field.validator,
                        )
                    else:
                        # Fallback for other field types - just use the original (may have bugs)
                        new_field = copy.copy(item_field)
                        new_field.key = compound_key
                    fields.append(new_field)
                # Add the "_more" prompt field for asking if user wants to add more
                more_field = ChoiceField(
                    key=f"{field.key}._more",
                    prompt=field.add_another_prompt,
                    description=f"Add another {field.description}",
                    choices=["yes", "no"],
                    case_sensitive=False,
                )
                self._field_to_list[more_field.key] = field.key
                fields.append(more_field)
            elif isinstance(field, PolymorphicListField):
                self._list_fields.add(field.key)
                # Add the type selector field first
                type_key = f"{field.key}._type"
                self._field_to_list[type_key] = field.key
                # Clone the type field with compound key
                type_field_clone = copy.copy(field.type_field)
                type_field_clone.key = type_key
                fields.append(type_field_clone)

                # Add variant fields with compound keys (e.g., "auths.username", "auths.password")
                for variant_name, variant_fields in field.variants.items():
                    for item_field in variant_fields:
                        compound_key = f"{field.key}.{item_field.key}"
                        self._field_to_list[compound_key] = field.key
                        # Store which variant this field belongs to
                        self._field_to_group[compound_key] = variant_name
                        # Create a NEW instance with the compound key
                        new_field: FormField  # type: ignore[no-redef]
                        if isinstance(item_field, TextField):
                            new_field = TextField(
                                key=compound_key,
                                prompt=item_field.prompt,
                                description=item_field.description,
                                min_length=item_field.min_length,
                                max_length=item_field.max_length,
                                pattern=item_field.pattern,
                                skippable=item_field.skippable,
                                default=item_field.default,
                                validator=item_field.validator,
                            )
                        elif isinstance(item_field, IntField):
                            new_field = IntField(
                                key=compound_key,
                                prompt=item_field.prompt,
                                description=item_field.description,
                                min_value=item_field.min_value,
                                max_value=item_field.max_value,
                                skippable=item_field.skippable,
                                default=item_field.default,
                                validator=item_field.validator,
                            )
                        elif isinstance(item_field, ChoiceField):
                            new_field = ChoiceField(
                                key=compound_key,
                                prompt=item_field.prompt,
                                description=item_field.description,
                                choices=item_field.choices or [],
                                case_sensitive=getattr(item_field, "case_sensitive", False),
                                skippable=item_field.skippable,
                                default=item_field.default,
                                validator=item_field.validator,
                            )
                        else:
                            new_field = copy.copy(item_field)
                            new_field.key = compound_key
                        fields.append(new_field)

                # Add the "_more" prompt field
                more_field = ChoiceField(
                    key=f"{field.key}._more",
                    prompt=field.add_another_prompt,
                    description=f"Add another {field.description}",
                    choices=["yes", "no"],
                    case_sensitive=False,
                )
                self._field_to_list[more_field.key] = field.key
                fields.append(more_field)
            else:
                self._field_to_group[field.key] = None
                fields.append(field)
        return fields

    def _get_session(self, chat: Chat) -> dict:
        """Get session state by deriving from message history."""
        return self._derive_session_from_history(chat)

    def _get_next_field_key(self, current_key: str) -> str | None:
        """Get the next field key after current."""
        idx = find_field_index_by_key(self._flat_fields, current_key)
        if idx + 1 < len(self._flat_fields):
            next_key = self._flat_fields[idx + 1].key

            # Check if this is a polymorphic list transition
            current_list = self._field_to_list.get(current_key)
            next_list = self._field_to_list.get(next_key)
            if current_list and next_list and current_list == next_list:
                # Both fields belong to the same list
                current_variant = self._field_to_group.get(current_key)
                next_variant = self._field_to_group.get(next_key)
                # If transitioning between variants, go to _more instead
                if current_variant and next_variant and current_variant != next_variant:
                    # Find the _more field for this list
                    more_key = f"{current_list}._more"
                    if find_field_index_by_key(self._flat_fields, more_key) < len(self._flat_fields):
                        return more_key

            return next_key
        return None

    def _get_first_uncollected_optional_field(self, session: dict) -> str | None:
        """Get the first uncollected optional field, or None if all collected."""
        return get_first_uncollected_optional_field(
            self._flat_fields, self._required_fields, session["collected_values"]
        )

    def generate_response(self, chat: Chat, user_message: HumanMessage) -> TrackResponse:
        """Process user input and generate next response."""
        session = self._get_session(chat)
        text = user_message.text.strip()
        current_key = session["current_field_key"]

        # Handle /start command - just return prompt for first field
        if text == "/start":
            return self._get_next_prompt(chat, session)

        # Handle exit command - return current collected values as JSON
        if text.lower() == self.exit_command.lower():
            result_values = self._reconstruct_nested_values(session)
            # Apply polymorphic list conversion
            result_values = self._convert_polymorphic_lists(result_values)
            # Use model_construct to bypass validation for missing required fields
            result = self.model.model_construct(**result_values)
            return "final", result.model_dump_json()  # type: ignore[return-value]

        # Handle confirmation state
        confirmation_response = self._handle_confirmation_response(text, chat, session)
        if confirmation_response is not None:
            return confirmation_response

        # Find current field
        current_field = next((f for f in self._flat_fields if f.key == current_key), None)
        if current_field is None:
            return self._handle_completion(chat, session)

        # Handle skip command
        if text.lower() == self.skip_command.lower():
            if is_field_required(self._required_fields, current_key) and not current_field.skippable:
                return current_key, f"{current_field.prompt}\nThis field is required and cannot be skipped."  # type: ignore[return-value]
            return self._handle_skip_command(current_key, current_field, session, chat)  # type: ignore[return-value]

        # Validate and parse input
        is_valid, error_msg, parsed_value = self._validate_and_parse_field(current_field, text)
        if not is_valid:
            return current_key, f"{current_field.prompt}\n{error_msg}"  # type: ignore[return-value]

        # Store value and move to next field
        # Check if this field belongs to a list
        list_key = self._field_to_list.get(current_key)

        if list_key and not current_key.endswith("._more"):
            _next_key, response = self._handle_list_field_value(current_key, parsed_value, list_key, session)
            if response is not None:
                return response
            # Normal flow for non-polymorphic list fields
            self._handle_regular_field_storage(current_key, parsed_value, session)
            return self._get_next_prompt(chat, session)
        elif current_key.endswith("._more"):
            # Handle "add another" prompt
            if list_key:
                return self._handle_more_field(list_key, str(parsed_value), current_key, session, chat)
        else:
            # Regular field
            self._handle_regular_field_storage(current_key, parsed_value, session)

        return self._get_next_prompt(chat, session)

    def _get_next_prompt(self, chat: Chat, session: dict) -> TrackResponse:
        """Get prompt for next field or complete form."""
        current_key = session["current_field_key"]

        # Check if form is complete
        if not any(f.key == current_key for f in self._flat_fields):
            return self._handle_completion(chat, session)

        field = next((f for f in self._flat_fields if f.key == current_key), None)
        if field is None:
            return self._handle_completion(chat, session)

        # Check for default strategy
        if field.key in self.base_descriptor.default_strategy:
            source_key = self.base_descriptor.default_strategy[field.key]
            if source_key in session["collected_values"]:
                default_value = session["collected_values"][source_key]
                prompt = f"{field.prompt} (default: {default_value})"
                return current_key, prompt  # type: ignore[return-value]

        return current_key, field.prompt  # type: ignore[return-value]

    def _handle_completion(self, chat: Chat, session: dict) -> TrackResponse:
        """Handle form completion."""
        if self.confirm_at_end and not session.get("confirmed"):
            session["confirmed"] = True
            # Set a special marker for confirmation state
            session["current_field_key"] = CONFIRMATION_STATE
            summary = self._format_summary(session["collected_values"])
            return CONFIRMATION_STATE, f"Confirm? (yes/no)\n{summary}"

        # Build result and return as JSON
        result = self._get_result_and_reset(chat, session)
        return "final", result.model_dump_json()  # type: ignore[return-value]

    def _format_summary(self, values: dict[str, Any]) -> str:
        """Format collected values as summary."""
        lines = ["Collected data:"]
        for field in self._fields:
            if isinstance(field, GroupField):
                # Handle nested group
                nested_lines = []
                has_values = False
                for nested_field in field.fields:
                    value = values.get(nested_field.key)
                    if value is not None and value != "":
                        nested_lines.append(f"    {nested_field.key}: {value}")
                        has_values = True
                if has_values:
                    lines.append(f"  {field.key}:")
                    lines.extend(nested_lines)
                else:
                    lines.append(f"  {field.key}: (skipped)")
            else:
                value = values.get(field.key)
                if value is None or value == "":
                    lines.append(f"  {field.key}: (skipped)")
                else:
                    lines.append(f"  {field.key}: {value}")
        return "\n".join(lines)

    def _reconstruct_nested_values(self, session: dict) -> dict[str, Any]:
        """Reconstruct nested structure from collected values and list items."""
        result: dict[str, Any] = {}
        list_items = session.get("list_items", {})
        collected_values = session.get("collected_values", {})

        for field in self._fields:
            if isinstance(field, GroupField):
                # Collect nested field values
                nested_dict: dict[str, Any] = {}
                has_any_value = False
                for nested_field in field.fields:
                    value = collected_values.get(nested_field.key)
                    if value is not None and value != "":
                        nested_dict[nested_field.key] = value
                        has_any_value = True
                # Only include the group if at least one nested field has a value
                if has_any_value:
                    result[field.key] = nested_dict
            elif isinstance(field, ListField):
                # Collect list items
                items = list_items.get(field.key, [])
                # Add current item if it has values
                current_list_item = session.get("current_list_item", {}).get(field.key, {})
                if current_list_item:
                    items = [*items, current_list_item]
                if items or field.min_items == 0:
                    result[field.key] = items
            elif isinstance(field, PolymorphicListField):
                # Collect polymorphic list items
                items = list_items.get(field.key, [])
                # Add current item if it has values
                current_list_item = session.get("current_list_item", {}).get(field.key, {})
                if current_list_item:
                    items = [*items, current_list_item]
                if items or field.min_items == 0:
                    result[field.key] = items
            else:
                value = collected_values.get(field.key)
                if value is not None:
                    result[field.key] = value
        return result

    def _convert_polymorphic_lists(self, result_values: dict[str, Any]) -> dict[str, Any]:
        """Convert polymorphic list dicts to proper model instances."""
        for field in self._fields:
            if isinstance(field, PolymorphicListField) and field.key in result_values:
                items_data = result_values[field.key]
                field_info = self.model.model_fields.get(field.key)
                if field_info:
                    # Get the inner type from list[X] -> X
                    annotation = field_info.annotation
                    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
                    if origin is list and args:
                        # args[0] is the inner type (e.g., Annotated[AuthType, Field(...)])
                        inner_type = args[0]
                        # Unwrap Annotated if present to get the actual union type
                        if typing.get_origin(inner_type) is Annotated:
                            inner_type = typing.get_args(inner_type)[0]
                        # Use TypeAdapter with the union type - it should handle discriminated unions
                        type_adapter = pydantic.TypeAdapter(inner_type)
                        # Convert each item to the proper variant
                        converted_items = []
                        for item in items_data:
                            try:
                                converted_items.append(type_adapter.validate_python(item))
                            except Exception:
                                # Fallback - keep as dict
                                converted_items.append(item)
                        result_values[field.key] = converted_items
        return result_values

    def _get_result_and_reset(self, _chat: Chat, session: dict) -> BaseModel:
        """Build result from collected values."""
        result_values = self._reconstruct_nested_values(session)
        result_values = self._convert_polymorphic_lists(result_values)
        result = self.model(**result_values)
        return result

    def get_result(self, chat: Chat) -> BaseModel | None:
        """Get collected result for this chat."""
        session = self._derive_session_from_history(chat)
        collected_values = session.get("collected_values", {})
        list_items = session.get("list_items", {})
        if not collected_values and not any(list_items.values()):
            return None
        result_values = self._reconstruct_nested_values(session)
        result_values = self._convert_polymorphic_lists(result_values)
        return self.model(**result_values)


class GroupField(FormField):
    """A field containing nested fields."""

    def __init__(
        self,
        *,
        key: str,
        prompt: str,
        description: str,
        fields: list[FormField],
        skippable: bool = True,
    ):
        super().__init__(
            key=key,
            prompt=prompt,
            description=description,
            skippable=skippable,
        )
        self.fields = fields

    # Validation is handled by nested fields, so no override needed

    def parse(self, value: str) -> Any:
        # GroupField doesn't parse directly
        return value


class ListField(FormField):
    """A field that collects a list of items with nested fields."""

    def __init__(
        self,
        *,
        key: str,
        prompt: str,
        description: str,
        item_fields: list[FormField],
        skippable: bool = True,
        min_items: int = 0,
        max_items: int | None = None,
        add_another_prompt: str = "Add another item?",
    ):
        super().__init__(
            key=key,
            prompt=prompt,
            description=description,
            skippable=skippable,
        )
        self.item_fields = item_fields
        self.min_items = min_items
        self.max_items = max_items
        self.add_another_prompt = add_another_prompt

    # Validation is handled by item fields, so no override needed

    def parse(self, value: str) -> Any:
        # ListField doesn't parse directly
        return value


class PolymorphicListField(FormField):
    """A field that collects a list of items with different field sets per variant.

    Each item in the list can be a different type (variant) with its own fields.
    The user selects which variant to add, then fills in that variant's fields.
    """

    def __init__(
        self,
        *,
        key: str,
        prompt: str,
        description: str,
        type_field: FormField,
        variants: dict[str, list[FormField]],
        skippable: bool = True,
        min_items: int = 0,
        max_items: int | None = None,
        add_another_prompt: str = "Add another item?",
    ):
        super().__init__(
            key=key,
            prompt=prompt,
            description=description,
            skippable=skippable,
        )
        self.type_field = type_field
        self.variants = variants
        self.min_items = min_items
        self.max_items = max_items
        self.add_another_prompt = add_another_prompt

    # Validation is handled by item fields, so no override needed

    def parse(self, value: str) -> Any:
        # PolymorphicListField doesn't parse directly
        return value


# Individual field types (to be implemented)


class TextField(FormField):
    """Text input with optional length and pattern validation."""

    def __init__(
        self,
        *,
        key: str,
        prompt: str,
        description: str = "",
        min_length: int = 0,
        max_length: int | None = None,
        pattern: str | None = None,
        skippable: bool = True,
        default: str = "",
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
    ):
        super().__init__(
            key=key,
            prompt=prompt,
            description=description,
            skippable=skippable,
            default=default,
            validator=validator,
        )
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern

    def _default_validate(self, value: str) -> tuple[bool, str | None]:
        return validate_text(value, self.min_length, self.max_length, self.pattern)

    def parse(self, value: str) -> str:
        return value


class IntField(FormField):
    """Integer input with optional min/max validation."""

    def __init__(
        self,
        *,
        key: str,
        prompt: str,
        description: str = "",
        min_value: int | None = None,
        max_value: int | None = None,
        skippable: bool = True,
        default: int = 0,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
    ):
        super().__init__(
            key=key,
            prompt=prompt,
            description=description,
            skippable=skippable,
            default=default,
            validator=validator,
        )
        self.min_value = min_value
        self.max_value = max_value

    def _default_validate(self, value: str) -> tuple[bool, str | None]:
        return validate_int(value, self.min_value, self.max_value)

    def parse(self, value: str) -> int:
        return parse_int(value)


class FloatField(FormField):
    """Float input with optional min/max validation."""

    def __init__(
        self,
        *,
        key: str,
        prompt: str,
        description: str = "",
        min_value: float | None = None,
        max_value: float | None = None,
        skippable: bool = True,
        default: float = 0.0,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
    ):
        super().__init__(
            key=key,
            prompt=prompt,
            description=description,
            skippable=skippable,
            default=default,
            validator=validator,
        )
        self.min_value = min_value
        self.max_value = max_value

    def _default_validate(self, value: str) -> tuple[bool, str | None]:
        return validate_float(value, self.min_value, self.max_value)

    def parse(self, value: str) -> float:
        return parse_float(value)


class ChoiceField(FormField):
    """Single choice from a list of options."""

    def __init__(
        self,
        *,
        key: str,
        prompt: str,
        description: str = "",
        choices: list[str],
        case_sensitive: bool = False,
        skippable: bool = True,
        default: str | None = None,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
    ):
        super().__init__(
            key=key,
            prompt=prompt,
            description=description,
            skippable=skippable,
            default=default,
            validator=validator,
            choices=choices,
        )
        self.case_sensitive = case_sensitive

    def _default_validate(self, value: str) -> tuple[bool, str | None]:
        return validate_choice(value, self.choices, self.case_sensitive)

    def parse(self, value: str) -> str:
        return parse_choice(value, self.choices, self.case_sensitive)


class BoolField(FormField):
    """Boolean yes/no field."""

    def __init__(
        self,
        *,
        key: str,
        prompt: str,
        description: str = "",
        yes_values: set[str] | None = None,
        no_values: set[str] | None = None,
        skippable: bool = True,
        default: bool = False,
        validator: Callable[[str], tuple[bool, str | None]] | None = None,
    ):
        super().__init__(
            key=key,
            prompt=prompt,
            description=description,
            skippable=skippable,
            default=default,
            validator=validator,
        )
        self.yes_values = yes_values or {"yes", "y", "true", "1", "да", "д"}
        self.no_values = no_values or {"no", "n", "false", "0", "нет", "н"}

    def _default_validate(self, value: str) -> tuple[bool, str | None]:
        return validate_bool(value, self.yes_values, self.no_values)

    def parse(self, value: str) -> bool:
        return parse_bool(value, self.yes_values)
