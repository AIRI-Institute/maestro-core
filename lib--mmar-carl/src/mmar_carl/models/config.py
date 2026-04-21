"""
Step configuration models for CARL reasoning system.
"""

from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field

from .enums import MemoryOperation


class ExecutionMode(str, Enum):
    """Execution strategy for LLM step."""

    FAST = "fast"
    SELF_CRITIC = "self_critic"


class ToolParameter(BaseModel):
    """Parameter definition for tool calls."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(default="string", description="Parameter type (string, int, float, bool, list, dict)")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=True, description="Whether the parameter is required")
    default: Any = Field(default=None, description="Default value if not required")


class ToolStepConfig(BaseModel):
    """Configuration for tool/function call steps."""

    tool_name: str = Field(..., description="Name of the tool to call")
    tool_description: str = Field(default="", description="Description of what the tool does")
    parameters: list[ToolParameter] = Field(default_factory=list, description="Tool input parameters")
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Maps step context keys to tool parameter names. Use '$history' for previous results.",
    )
    output_key: str = Field(default="result", description="Key to store tool output in step result")
    timeout: float = Field(default=30.0, description="Timeout in seconds for tool execution")
    retry_on_error: bool = Field(default=True, description="Whether to retry on error")

    # The actual callable is set at runtime, not serialized
    _tool_callable: Optional[Callable] = None

    model_config = {"arbitrary_types_allowed": True}


class MCPServerConfig(BaseModel):
    """Configuration for MCP server connection."""

    server_name: str = Field(..., description="Name of the MCP server")
    transport: Literal["stdio", "http", "websocket"] = Field(default="stdio", description="Transport type")
    command: Optional[str] = Field(default=None, description="Command to start stdio server")
    args: list[str] = Field(default_factory=list, description="Arguments for stdio server")
    url: Optional[str] = Field(default=None, description="URL for HTTP/WebSocket server")
    headers: dict[str, str] = Field(default_factory=dict, description="Headers for HTTP/WebSocket")


class MCPStepConfig(BaseModel):
    """Configuration for MCP protocol steps."""

    server: MCPServerConfig = Field(..., description="MCP server configuration")
    tool_name: str = Field(..., description="Name of the MCP tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Static arguments for the tool")
    argument_mapping: dict[str, str] = Field(
        default_factory=dict, description="Maps step context keys to MCP tool arguments"
    )
    timeout: float = Field(default=60.0, description="Timeout in seconds")


class MemoryStepConfig(BaseModel):
    """Configuration for memory read/write steps."""

    operation: MemoryOperation = Field(..., description="Memory operation type")
    memory_key: str = Field(..., description="Key in memory store")
    value_source: Optional[str] = Field(
        default=None, description="Source of value for write operations (context key or '$history[-1]')"
    )
    default_value: Any = Field(default=None, description="Default value if key not found on read")
    namespace: str = Field(default="default", description="Memory namespace for isolation")


class TransformStepConfig(BaseModel):
    """Configuration for data transformation steps (no LLM call)."""

    transform_type: Literal["extract", "format", "aggregate", "filter", "map"] = Field(
        ..., description="Type of transformation"
    )
    input_key: str = Field(default="$history[-1]", description="Key to get input from")
    output_format: Optional[str] = Field(default=None, description="Output format template")
    expression: Optional[str] = Field(default=None, description="Transformation expression (for extract/filter)")
    # For 'map' operations - applies a simple template to each item
    map_template: Optional[str] = Field(default=None, description="Template for map operations")


class ConditionalBranch(BaseModel):
    """A single conditional branch."""

    condition: str = Field(..., description="Condition expression (evaluated against context)")
    next_step: int = Field(..., description="Step number to execute if condition is true")


class ConditionalStepConfig(BaseModel):
    """Configuration for conditional branching steps."""

    branches: list[ConditionalBranch] = Field(..., description="List of conditional branches")
    default_step: Optional[int] = Field(default=None, description="Default step if no condition matches")
    condition_context_key: str = Field(
        default="$history[-1]", description="Context key to evaluate conditions against"
    )


class StructuredOutputStepConfig(BaseModel):
    """Configuration for schema-constrained structured output steps."""

    input_source: str = Field(
        default="$history[-1]",
        description="Input source used to build the structured output prompt",
    )
    output_schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema that model output must match",
    )
    schema_name: str = Field(default="StructuredOutput", description="Human-readable schema name")
    instruction: str = Field(
        default="",
        description="Additional instruction for the model before schema conversion",
    )
    strict_json: bool = Field(
        default=True,
        description="If true, executor asks model to return only raw JSON",
    )

    @classmethod
    def from_pydantic_model(
        cls,
        model_cls: type[BaseModel],
        input_source: str = "$history[-1]",
        instruction: str = "",
        strict_json: bool = True,
    ) -> "StructuredOutputStepConfig":
        """Create structured config from a Pydantic model class."""
        return cls(
            input_source=input_source,
            output_schema=model_cls.model_json_schema(),
            schema_name=model_cls.__name__,
            instruction=instruction,
            strict_json=strict_json,
        )


# Union type for all step configurations
StepConfig = Union[
    ToolStepConfig,
    MCPStepConfig,
    MemoryStepConfig,
    TransformStepConfig,
    ConditionalStepConfig,
    StructuredOutputStepConfig,
    None,
]


class ContextQuery(BaseModel):
    """
    Individual context query with optional search configuration override.

    Allows fine-grained control over search strategy for specific queries.
    """

    query: str = Field(..., description="The query text for context extraction")
    search_strategy: Optional[Literal["substring", "vector"]] = Field(
        default=None, description="Override search strategy for this query"
    )
    search_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional search configuration for this query"
    )

    def __str__(self) -> str:
        return self.query


class LLMStepConfig(BaseModel):
    """
    Configuration for per-step LLM overrides.

    Allows specifying a different model for specific LLM steps,
    overriding the default set in ReasoningContext.

    Example usage:
        ```python
        # Override model for a specific step (OpenAI-compatible APIs)
        LLMStepDescription(
            number=1,
            title="Complex Analysis",
            aim="Perform complex analysis",
            llm_config=LLMStepConfig(model="anthropic/claude-3.5-sonnet")
        )
        ```
    """

    # Model override (for OpenAI-compatible APIs)
    model: Optional[str] = Field(
        default=None,
        description="Model identifier to use for this step (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet')",
    )

    # Temperature override
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperature for this step (overrides default)",
    )

    # Max tokens override
    max_tokens: Optional[int] = Field(
        default=None,
        description="Max tokens for this step (overrides default)",
    )

    # Per-step timeout override
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Timeout for this step in seconds (None = use chain default)",
    )

    # Execution mode override
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.FAST,
        description="Execution strategy for this step",
    )

    # SELF_CRITIC configuration
    self_critic_evaluators: list[str] = Field(
        default_factory=lambda: ["llm"],
        min_length=1,
        description="Ordered evaluator names for SELF_CRITIC mode; all must approve",
    )
    self_critic_max_revisions: int = Field(
        default=1,
        ge=0,
        description="Maximum candidate regeneration rounds when evaluator chain disapproves",
    )
    self_critic_instruction: str = Field(
        default="",
        description="Optional extra instruction for built-in 'llm' self-critic evaluator.",
    )
    self_critic_disapprove_feedback: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Static feedback appended to regeneration notes when evaluator disapproves. "
            "Keys are evaluator names; optional '*' applies to any evaluator."
        ),
    )
