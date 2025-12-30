"""User profile track for collecting name, email, and favorite fruits."""

from pydantic import BaseModel

from chat_manager_examples.beta_slot_filling_track import (
    TEXT_FIELD_PATTERNS,
    BaseFieldDescriptor,
    ListField,
    SlotFillingTrack,
    TextField,
)
from chat_manager_examples.config import DOMAINS


class FillingUserProfile(BaseModel):
    """User profile with name, email, and optional favorite fruits."""

    name: str
    email: str
    favorite_fruits: list[str] | None = None


# Field definitions for FillingUserProfile
NAME_FIELD = TextField(
    key="name",
    prompt="What's your name?",
    description="Full name",
    min_length=1,
)

EMAIL_FIELD = TextField(
    key="email",
    prompt="What's your email?",
    description="Email address",
    pattern=TEXT_FIELD_PATTERNS.email,
)

# Simple text field for fruit names (not nested like in the test)
FRUIT_NAME_FIELD = TextField(
    key="name",
    prompt="What's the fruit name?",
    description="Fruit name",
    min_length=1,
)

# ListField for collecting favorite fruits
FAVORITE_FRUITS_FIELD = ListField(
    key="favorite_fruits",
    prompt="Let's add your favorite fruits",
    description="List of favorite fruits",
    item_fields=[FRUIT_NAME_FIELD],
    add_another_prompt="Add another fruit?",
)

FILLING_USER_PROFILE_FIELDS = [NAME_FIELD, EMAIL_FIELD, FAVORITE_FRUITS_FIELD]


class FillingUserProfileTrack(SlotFillingTrack):
    """Track for collecting user profile with name, email, and optional favorite fruits."""

    DOMAIN = DOMAINS.examples
    CAPTION = "👤 Filling User Profile"

    def __init__(self):
        super().__init__(
            model=FillingUserProfile,
            fields=FILLING_USER_PROFILE_FIELDS,
            base_descriptor=BaseFieldDescriptor(),
            confirm_at_end=False,
        )

    def _reconstruct_nested_values(self, session: dict) -> dict:
        """Reconstruct nested values and convert favorite_fruits from dicts to strings."""
        result = super()._reconstruct_nested_values(session)

        # Convert favorite_fruits from list of dicts to list of strings
        if result.get("favorite_fruits"):
            result["favorite_fruits"] = [item.get("name", "") for item in result["favorite_fruits"] if item.get("name")]

        return result
