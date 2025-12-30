# Form Field - Design Document

## Overview
Design and implement a **FormField** for conversational data collection through chat interfaces. Users are guided through a series of fields, with validation and re-prompting on errors.

## Core Design

### Approach: Pydantic-First

Pass a Pydantic model to `FormTrack.__init__`. Fields are extracted from the model, with optional custom descriptors for prompts, validators, etc.

### FormTrack Constructor

```python
class FormTrack(SimpleTrack):
    def __init__(
        self,
        model: type[BaseModel],
        *,
        base_descriptor: BaseFieldDescriptor | None = None,
        field_descriptors: dict[str, FieldDescriptor] | None = None,
        confirm_at_end: bool = False,
    ):
        """
        Args:
            model: Pydantic BaseModel class defining the target structure
            base_descriptor: Default settings for all fields (language, commands, etc.)
            field_descriptors: Optional overrides for prompts, validators, etc.
            confirm_at_end: Show summary and require confirmation
        """
```

### FieldDescriptor

```python
from pydantic import BaseModel, Field

class FieldDescriptor(BaseModel):
    """Overrides for default field behavior."""
    prompt: str | None = None
    description: str | None = None
    extract_key: str | None = None  # For document extraction
    skippable: bool | None = None
    choices: list[str] | None = None  # For ChoiceField

    # Validator is a callable, excluded from model serialization
    validator: Callable[[Any], tuple[bool, str | None]] | None = Field(default=None, exclude=True)
```

### BaseFieldDescriptor

```python
class BaseFieldDescriptor(BaseModel):
    """Default settings applied to all fields unless overridden."""
    allow_skip: bool = True  # Allow skipping optional fields
    exit_command: str = "exit"
    skip_command: str = "skip"

    # Maps target field key -> source field key
    # When source field is filled, use its value as default for target field
    default_strategy: dict[str, str] = Field(default_factory=dict)
```

### Default Strategy

**Use case**: Auto-fill fields with values from previous fields (simple copy).

```python
base = BaseFieldDescriptor(
    default_strategy={
        "billing_address": "shipping_address",  # Copy shipping address
        "confirm_email": "email",  # Copy email for confirmation
        "username": "email",  # Use email as username
    },
)
```

**Flow:**
- When user fills `shipping_address` with "123 Main St"
- Before prompting for `billing_address`, use that value as default
- Show prompt with pre-filled value: "Billing address (default: 123 Main St):"
- User can accept default (press enter), type new value, or skip

**For transformations**: Add a `FieldDescriptor` with a custom validator that transforms the input:

```python
def slugify_transform(value: str) -> str:
    """Transform input to slug on first keystroke."""
    return value.lower().replace(" ", "-").replace("_", "-")

descriptors = {
    "project_slug": FieldDescriptor(
        prompt="Project slug",
        # User types project name, we auto-transform it
    ),
}

# Or pre-compute in code before creating track
```

### Default Field Inference

Fields are automatically inferred from Pydantic model:

| Pydantic Type | Inferred Field Type |
|---------------|---------------------|
| `str` | TextField |
| `int` | IntField |
| `float` | FloatField |
| `bool` | BoolField |
| `Literal["a", "b"]` | ChoiceField |
| `str` with "email" in field name | TextField with email pattern |
| `str` with "phone"/"tel" in field name | TextField with phone pattern |
| Nested `BaseModel` | GroupField |

### Example Usage

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class UserProfile(BaseModel):
    name: str = Field(description="Full name", min_length=2)
    age: int = Field(description="Age in years", ge=0, le=120)
    email: str
    favorite_color: Literal["red", "green", "blue", "yellow"] | None = None

    class Address(BaseModel):
        street: str = Field(description="Street address")
        city: str
        zip_code: str = Field(description="Postal code")

    address: Address | None = None

# Basic usage - all prompts auto-generated from descriptions
track = FormTrack(model=UserProfile)

# With base descriptor for custom commands
base = BaseFieldDescriptor(
    exit_command="выход",
    skip_command="пропустить",
)

# With custom prompts and validators
descriptors = {
    "name": FieldDescriptor(prompt="Как вас зовут?"),
    "age": FieldDescriptor(
        prompt="Сколько вам лет?",
        validator=lambda v: (True, None) if 18 <= v <= 100 else (False, "Must be 18-100")
    ),
    "email": FieldDescriptor(prompt="Ваша электронная почта?"),
    # Nested field - use compound key for the group
    "address": FieldDescriptor(
        prompt="Введите адрес (или отправьте документ)",
        extract_key="address",
    ),
    # Nested fields - use compound keys for individual fields
    "address.street": FieldDescriptor(prompt="Улица и номер дома"),
    "address.city": FieldDescriptor(prompt="Город"),
    "address.zip_code": FieldDescriptor(prompt="Почтовый индекс"),
}

track = FormTrack(
    model=UserProfile,
    base_descriptor=base,
    field_descriptors=descriptors,
    confirm_at_end=True,
)
```

### Auto-Generated Prompts

If no custom prompt is provided, generate from field info:

```python
# From Field(description="Email address")
prompt = "Please enter your email address"

# From field name
prompt = "Please enter your favorite_color"

# With constraints
prompt = "Please enter your age (must be between 0 and 120)"
```

### Conversation Flow

```
Bot: What's your name?
User: Alice

Bot: How old are you?
User: twenty-five

Bot: Please enter a valid number between 0 and 120.
Bot: How old are you?
User: 25

Bot: What's your email?
User: alice@example.com

Bot: Favorite color? (red/green/blue/yellow, or skip)
User: red

Bot: Enter your address (or send document)
Bot: Street:
User: 123 Main St

Bot: City:
User: Boston

Bot: ZIP code:
User: 02101

Bot: Form complete!
    Name: Alice
    Age: 25
    Email: alice@example.com
    Favorite color: red
    Address:
      Street: 123 Main St
      City: Boston
      ZIP code: 02101
```

### State Management

Internal state per session:
- `current_field_key`: key of the field we're currently on (or nested field key like "address.street")
- `collected_values`: dict of filled values (flat structure with compound keys for nested fields)
- `last_error`: validation error message (for re-prompting)

**Flattening**: Nested BaseModel fields are flattened using compound keys (e.g., "address.street", "address.city").

**Index lookup**: On each request, find `current_field_index` by searching for `current_field_key` in the flattened fields list.

### Validation

Two sources of validation:

**Pydantic validation** (always applied):
- Type checking
- Field constraints (ge, le, min_length, max_length, pattern)
- Custom @field_validator functions

**Additional FieldDescriptor.validator** (optional):
- Extra custom validation beyond Pydantic
- Returns `(is_valid, error_message)`
- Runs after Pydantic validation

### Error Handling

| Error | Handler |
|-------|---------|
| Type mismatch (string for int) | Pydantic error + re-ask |
| Out of range (age > 120) | Pydantic error + re-ask |
| Invalid choice | Pydantic error + re-ask |
| FieldDescriptor.validator fail | Custom error + re-ask |
| Required field skipped | Re-ask with "This field is required" |
| Optional field skipped | Use default/None, move to next field |
| Pattern mismatch | Pydantic error + re-ask |

### Output Format

Returns instance of the Pydantic model with nested structure preserved:

```python
result = track.get_result(session_id)
# UserProfile(
#     name="Alice",
#     age=25,
#     email="alice@example.com",
#     favorite_color="red",
#     address=Address(
#         street="123 Main St",
#         city="Boston",
#         zip_code="02101"
#     )
# )
```

### Document Extraction Scenario

**Use case**: User is filling address information but sends a scan of their ID/utility bill instead of typing.

```
Bot: Street:
User: [sends image: id_card.jpg]
```

**Flow:**
- Detect `user_message.resource_id` is present
- Pass to `DocumentExtractorAPI.extract(resource_id)`
- Get structured JSON back with extracted fields
- Map extracted fields to model fields using `extract_key` from FieldDescriptor (or inferred from field name)
- Validate each extracted value with Pydantic + custom validators
- Present summary: "Found: Street=123 Main St, City=Boston. Use this data? (yes/edit/manual)"

**FieldDescriptor for extraction:**
```python
field_descriptors = {
    "address": FieldDescriptor(
        extract_key="address",  # Top-level key in document JSON
        allow_extraction=True,   # Enable extraction for this group
    ),
    "address.street": FieldDescriptor(
        extract_key="address.street",  # Specific path in document JSON
    ),
}
```

**Automatic extraction key inference:**
- Field `name` → looks for `"name"` in document JSON
- Field `address.street` → looks for `"address.street"` or nested `{"address": {"street": ...}}`

### Open Questions

- Multi-level nesting: Should nested BaseModels support further nesting?
- Edit mode: Allow users to go back and edit previous fields?
- Document extraction confidence: What if extraction is uncertain? Show confidence score?
- Extraction retry: If extraction fails/partial, re-ask or allow manual entry?
- Complex types: How to handle List, Dict, Union types?
- Optional fields with default: Skip or prompt with default shown?

### Implementation Checklist

- BaseFieldDescriptor as BaseModel (commands, default_strategy)
- FieldDescriptor as BaseModel (per-field overrides)
- Default strategy: copy field defaults from other fields
- FormField base class (internal, not exposed)
- TextField, IntField, FloatField, ChoiceField, BoolField
- TEXT_FIELD_PATTERNS with predefined patterns (email, phone)
- GroupField for nested BaseModel
- FormTrack model field introspection
- FormTrack field type inference from Pydantic types
- FormTrack field flattening with compound keys
- FormTrack index lookup by key
- FormTrack session management
- FormTrack prompt generation (from description/field name)
- FormTrack base_descriptor field defaults
- FormTrack default strategy resolution and pre-fill
- Pydantic validation integration
- Custom FieldDescriptor.validator integration
- Summary formatting with nesting
- Skip/exit command handling (from base_descriptor)
- Resource_id detection for document extraction
- DocumentExtractorAPI integration
- Extract confirmation flow (yes/edit/manual)
- Type: ignore for mypy

### Compound Key System

For nested BaseModel fields:
- Flattened key: `"address.street"`, `"address.city"`, `"address.zip"`
- Session stores: `{"address.street": "123 Main", "address.city": "Boston"}`
- Reconstruct to: `Address(street="123 Main", city="Boston")`
