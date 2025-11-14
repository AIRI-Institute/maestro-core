"""mmar-utils package.

Utilities for multi-modal architectures team
"""

from mmar_utils.models import ResourcesModel
from mmar_utils.validators import ExistingDir

from .decorators_on_error_log_and_none import on_error_log_and_none
from .decorators_retry import retry_on_ex, retry_on_cond
from .decorators_trace_with import FunctionCall, FunctionEnter, FunctionInvocation, trace_with
from .mmar_types import Either
from .parallel_map import parallel_map
from .utils import read_json, try_parse_bool, try_parse_float, try_parse_int, try_parse_json, noop, anoop
from .utils_collections import edit_object, flatten, take_exactly_one
from .utils_texts import (
    chunk_respect_semantic,
    extract_text_inside,
    pretty_line,
    remove_prefix_if_present,
    remove_suffix_if_present,
    rindex_safe,
)
from .utils_texts_postprocessing import clean_and_fix_text, postprocess_text
from .validators import ExistingFile, ExistingPath, Message, Prompt, PromptBaseContext, SecretStrNotEmpty, StrNotEmpty, validate_no_underscores

__all__ = [
    "Either",
    "ExistingDir",
    "ExistingFile",
    "ExistingPath",
    "FunctionCall",
    "FunctionEnter",
    "FunctionInvocation",
    "Message",
    "Prompt",
    "PromptBaseContext",
    "SecretStrNotEmpty",
    "StrNotEmpty",
    "chunk_respect_semantic",
    "edit_object",
    "extract_text_inside",
    "flatten",
    "on_error_log_and_none",
    "parallel_map",
    "pretty_line",
    "read_json",
    "remove_prefix_if_present",
    "remove_suffix_if_present",
    "retry_on_cond",
    "retry_on_ex",
    "rindex_safe",
    "trace_with",
    "try_parse_bool",
    "try_parse_float",
    "try_parse_int",
    "try_parse_json",
    "postprocess_text",
    "ResourcesModel",
    "clean_and_fix_text",
    "noop",
    "anoop",
    "validate_no_underscores",
    "take_exactly_one",
]
