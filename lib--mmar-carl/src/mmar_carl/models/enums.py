"""
Enums for CARL reasoning system.
"""

from enum import StrEnum


class StepType(StrEnum):
    """Types of steps supported in reasoning chains."""

    LLM = "llm"  # Standard LLM reasoning step (default)
    TOOL = "tool"  # External tool/function call
    MCP = "mcp"  # Model Context Protocol server call
    MEMORY = "memory"  # Memory read/write operation
    TRANSFORM = "transform"  # Data transformation (no LLM)
    CONDITIONAL = "conditional"  # Conditional branching
    STRUCTURED_OUTPUT = "structured_output"  # LLM step with schema-constrained JSON output


class MemoryOperation(StrEnum):
    """Types of memory operations."""

    READ = "read"
    WRITE = "write"
    APPEND = "append"
    DELETE = "delete"
    LIST = "list"


class Language(StrEnum):
    """Supported languages."""

    RUSSIAN = "ru"
    ENGLISH = "en"
