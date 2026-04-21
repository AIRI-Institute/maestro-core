"""
Step description classes for CARL reasoning system.
"""

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import (
    ConditionalStepConfig,
    ContextQuery,
    LLMStepConfig,
    MCPStepConfig,
    MemoryStepConfig,
    StepConfig,
    ToolStepConfig,
    TransformStepConfig,
    StructuredOutputStepConfig,
)
from .enums import StepType


class StepDescriptionBase(BaseModel):
    """
    Abstract base class for all step descriptions.

    All step types share these common fields and methods.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # === Core Fields (required for all step types) ===
    number: int = Field(..., description="Step number in the sequence")
    title: str = Field(..., description="Human-readable title of the step")
    dependencies: list[int] = Field(default_factory=list, description="List of step numbers this step depends on")
    checkpoint: bool = Field(default=False, description="Mark this step as a RE-PLAN rollback checkpoint")
    checkpoint_name: str | None = Field(default=None, description="Optional checkpoint name")
    replan_enabled: bool | None = Field(
        default=None,
        description="Optional per-step RE-PLAN override (None = chain policy default)",
    )
    metrics: list[Any] = Field(
        default_factory=list,
        description="List of MetricBase instances to evaluate after step execution",
        exclude=True,
    )

    # Step type is determined by the concrete class
    @property
    def step_type(self) -> StepType:
        """Get the step type. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement step_type")

    def depends_on(self, step_number: int) -> bool:
        """Check if this step depends on a given step number."""
        return step_number in self.dependencies

    def has_dependencies(self) -> bool:
        """Check if this step has any dependencies."""
        return len(self.dependencies) > 0

    def is_llm_step(self) -> bool:
        """Check if this is an LLM reasoning step."""
        return self.step_type == StepType.LLM

    def is_tool_step(self) -> bool:
        """Check if this is a tool execution step."""
        return self.step_type == StepType.TOOL

    def is_mcp_step(self) -> bool:
        """Check if this is an MCP protocol step."""
        return self.step_type == StepType.MCP

    def is_memory_step(self) -> bool:
        """Check if this is a memory operation step."""
        return self.step_type == StepType.MEMORY

    def is_transform_step(self) -> bool:
        """Check if this is a data transformation step."""
        return self.step_type == StepType.TRANSFORM

    def is_conditional_step(self) -> bool:
        """Check if this is a conditional branching step."""
        return self.step_type == StepType.CONDITIONAL

    def is_structured_output_step(self) -> bool:
        """Check if this is a structured output generation step."""
        return self.step_type == StepType.STRUCTURED_OUTPUT

    # For serialization - subclasses should provide step_config if applicable
    @property
    def step_config(self) -> Optional[StepConfig]:
        """Get step-specific configuration. Override in subclasses."""
        return None

    def get_llm_field(self, field_name: str, default: str = "") -> str:
        """
        Safely get an LLM-specific field value.

        Works for both typed LLMStepDescription and legacy StepDescription.
        Returns default for non-LLM step types.
        """
        return getattr(self, field_name, default)


class LLMStepDescription(StepDescriptionBase):
    """
    LLM reasoning step description.

    This is the default step type for chain-of-thought reasoning with LLM.

    Supports per-step LLM configuration via the llm_config field:
        ```python
        LLMStepDescription(
            number=1,
            title="Complex Analysis",
            aim="Perform deep analysis",
            llm_config=LLMStepConfig(model="anthropic/claude-3.5-sonnet")
        )
        ```
    """

    # LLM-specific fields
    aim: str = Field(default="", description="Primary objective of this step")
    reasoning_questions: str = Field(default="", description="Key questions to answer")
    step_context_queries: list[ContextQuery | str] = Field(
        default_factory=list,
        description="List of queries to extract relevant context from outer_context (RAG-like)",
    )
    stage_action: str = Field(default="", description="Specific action to perform")
    example_reasoning: str = Field(default="", description="Example of expert reasoning")

    # Per-step LLM configuration override
    llm_config: Optional[LLMStepConfig] = Field(
        default=None,
        description="Optional LLM configuration override for this step (model, temperature, etc.)",
    )

    # Per-step retry override
    retry_max: Optional[int] = Field(
        default=None,
        description="Override retry attempts for this step (None = use context default)",
    )

    # Per-step timeout override
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Timeout for this step in seconds (None = use chain default)",
    )

    @property
    def step_type(self) -> StepType:
        return StepType.LLM

    @model_validator(mode="after")
    def validate_llm_fields(self) -> "LLMStepDescription":
        """Validate that LLM step has required fields."""
        if not self.aim:
            raise ValueError("LLM steps require 'aim' to be set")
        return self

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


class ToolStepDescription(StepDescriptionBase):
    """
    Tool execution step description.

    Executes external functions/tools registered in the context.
    """

    config: ToolStepConfig = Field(..., description="Tool configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.TOOL

    @property
    def step_config(self) -> ToolStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


class MCPStepDescription(StepDescriptionBase):
    """
    MCP (Model Context Protocol) step description.

    Calls tools on MCP servers.
    """

    config: MCPStepConfig = Field(..., description="MCP configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.MCP

    @property
    def step_config(self) -> MCPStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


class MemoryStepDescription(StepDescriptionBase):
    """
    Memory operation step description.

    Performs read/write/append/delete/list operations on shared memory.
    """

    config: MemoryStepConfig = Field(..., description="Memory operation configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.MEMORY

    @property
    def step_config(self) -> MemoryStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


class TransformStepDescription(StepDescriptionBase):
    """
    Data transformation step description.

    Performs data transformations without LLM calls.
    """

    config: TransformStepConfig = Field(..., description="Transform configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.TRANSFORM

    @property
    def step_config(self) -> TransformStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


class ConditionalStepDescription(StepDescriptionBase):
    """
    Conditional branching step description.

    Evaluates conditions and determines next step.
    """

    config: ConditionalStepConfig = Field(..., description="Conditional configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.CONDITIONAL

    @property
    def step_config(self) -> ConditionalStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


class StructuredOutputStepDescription(StepDescriptionBase):
    """Structured output generation step description."""

    config: StructuredOutputStepConfig = Field(..., description="Structured output configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.STRUCTURED_OUTPUT

    @property
    def step_config(self) -> StructuredOutputStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


# Union type for all step descriptions
AnyStepDescription = Union[
    LLMStepDescription,
    ToolStepDescription,
    MCPStepDescription,
    MemoryStepDescription,
    TransformStepDescription,
    ConditionalStepDescription,
    StructuredOutputStepDescription,
]


class StepDescription(BaseModel):
    """
    DEPRECATED: Use typed step classes instead.

    This class provides BACKWARD COMPATIBILITY with the previous unified API.
    For new code, prefer using the specific step classes:
    - LLMStepDescription - for LLM reasoning steps
    - ToolStepDescription - for external function calls
    - MCPStepDescription - for MCP protocol calls
    - MemoryStepDescription - for memory operations
    - TransformStepDescription - for data transformations
    - ConditionalStepDescription - for conditional branching
    - StructuredOutputStepDescription - for structured JSON output

    Migration example:
        # Old (deprecated):
        StepDescription(
            number=1,
            title="Analysis",
            aim="Analyze data",
            step_type=StepType.LLM
        )

        # New (recommended):
        LLMStepDescription(
            number=1,
            title="Analysis",
            aim="Analyze data"
        )

    Supports multiple step types:
    - LLM (default): Standard LLM reasoning with prompts
    - TOOL: External function/tool execution
    - MCP: Model Context Protocol server calls
    - MEMORY: Read/write operations on shared memory
    - TRANSFORM: Data transformations without LLM
    - CONDITIONAL: Branching logic based on conditions
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initialize with deprecation warning."""
        import warnings

        warnings.warn(
            "StepDescription is deprecated. Use typed step classes instead: "
            "LLMStepDescription, ToolStepDescription, MCPStepDescription, "
            "MemoryStepDescription, TransformStepDescription, ConditionalStepDescription. "
            "See documentation for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)

    # === Core Fields (required for all step types) ===
    number: int = Field(..., description="Step number in the sequence")
    title: str = Field(..., description="Human-readable title of the step")
    dependencies: list[int] = Field(default_factory=list, description="List of step numbers this step depends on")
    checkpoint: bool = Field(default=False, description="Mark this step as a RE-PLAN rollback checkpoint")
    checkpoint_name: str | None = Field(default=None, description="Optional checkpoint name")
    replan_enabled: bool | None = Field(
        default=None,
        description="Optional per-step RE-PLAN override (None = chain policy default)",
    )

    # === Step Type Configuration ===
    step_type: StepType = Field(default=StepType.LLM, description="Type of step execution")
    step_config: Optional[
        Union[
            ToolStepConfig,
            MCPStepConfig,
            MemoryStepConfig,
            TransformStepConfig,
            ConditionalStepConfig,
            StructuredOutputStepConfig,
        ]
    ] = Field(default=None, description="Type-specific configuration (required for non-LLM steps)")

    # === LLM Step Fields (used when step_type=LLM) ===
    aim: str = Field(default="", description="Primary objective of this step")
    reasoning_questions: str = Field(default="", description="Key questions to answer")
    step_context_queries: list[ContextQuery | str] = Field(
        default_factory=list, description="List of queries to extract relevant context from outer_context (RAG-like)"
    )
    stage_action: str = Field(default="", description="Specific action to perform")
    example_reasoning: str = Field(default="", description="Example of expert reasoning")

    # Per-step LLM configuration override (used when step_type=LLM)
    llm_config: Optional[LLMStepConfig] = Field(
        default=None,
        description="Optional LLM configuration override for this step (model, temperature, etc.)",
    )

    # Per-step retry override
    retry_max: Optional[int] = Field(
        default=None,
        description="Override retry attempts for this step (None = use context default)",
    )

    # Per-step timeout override
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Timeout for this step in seconds (None = use chain default)",
    )

    metrics: list[Any] = Field(
        default_factory=list,
        description="List of MetricBase instances to evaluate after step execution",
        exclude=True,
    )

    @model_validator(mode="after")
    def validate_step_config(self) -> "StepDescription":
        """Validate that step configuration matches step type."""
        if self.step_type == StepType.LLM:
            # LLM steps need aim at minimum
            if not self.aim:
                raise ValueError("LLM steps require 'aim' to be set")
        elif self.step_type == StepType.TOOL:
            if not isinstance(self.step_config, ToolStepConfig):
                raise ValueError("TOOL steps require ToolStepConfig")
        elif self.step_type == StepType.MCP:
            if not isinstance(self.step_config, MCPStepConfig):
                raise ValueError("MCP steps require MCPStepConfig")
        elif self.step_type == StepType.MEMORY:
            if not isinstance(self.step_config, MemoryStepConfig):
                raise ValueError("MEMORY steps require MemoryStepConfig")
        elif self.step_type == StepType.TRANSFORM:
            if not isinstance(self.step_config, TransformStepConfig):
                raise ValueError("TRANSFORM steps require TransformStepConfig")
        elif self.step_type == StepType.CONDITIONAL:
            if not isinstance(self.step_config, ConditionalStepConfig):
                raise ValueError("CONDITIONAL steps require ConditionalStepConfig")
        elif self.step_type == StepType.STRUCTURED_OUTPUT:
            if not isinstance(self.step_config, StructuredOutputStepConfig):
                raise ValueError("STRUCTURED_OUTPUT steps require StructuredOutputStepConfig")
        return self

    def depends_on(self, step_number: int) -> bool:
        """Check if this step depends on a given step number."""
        return step_number in self.dependencies

    def has_dependencies(self) -> bool:
        """Check if this step has any dependencies."""
        return len(self.dependencies) > 0

    def is_llm_step(self) -> bool:
        """Check if this is an LLM reasoning step."""
        return self.step_type == StepType.LLM

    def is_tool_step(self) -> bool:
        """Check if this is a tool execution step."""
        return self.step_type == StepType.TOOL

    def is_mcp_step(self) -> bool:
        """Check if this is an MCP protocol step."""
        return self.step_type == StepType.MCP

    def is_memory_step(self) -> bool:
        """Check if this is a memory operation step."""
        return self.step_type == StepType.MEMORY

    def is_transform_step(self) -> bool:
        """Check if this is a data transformation step."""
        return self.step_type == StepType.TRANSFORM

    def is_conditional_step(self) -> bool:
        """Check if this is a conditional branching step."""
        return self.step_type == StepType.CONDITIONAL

    def is_structured_output_step(self) -> bool:
        """Check if this is a structured output generation step."""
        return self.step_type == StepType.STRUCTURED_OUTPUT

    def to_typed_step(self) -> AnyStepDescription:
        """
        Convert this legacy StepDescription to the appropriate typed step class.

        Returns:
            The appropriate typed step description instance.
        """
        if self.step_type == StepType.LLM:
            return LLMStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                aim=self.aim,
                reasoning_questions=self.reasoning_questions,
                step_context_queries=self.step_context_queries,
                stage_action=self.stage_action,
                example_reasoning=self.example_reasoning,
                llm_config=self.llm_config,
            )
        elif self.step_type == StepType.TOOL:
            return ToolStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.MCP:
            return MCPStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.MEMORY:
            return MemoryStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.TRANSFORM:
            return TransformStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.CONDITIONAL:
            return ConditionalStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.STRUCTURED_OUTPUT:
            return StructuredOutputStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        else:
            raise ValueError(f"Unknown step type: {self.step_type}")


def create_step(
    number: int,
    title: str,
    step_type: StepType = StepType.LLM,
    dependencies: list[int] | None = None,
    checkpoint: bool = False,
    checkpoint_name: str | None = None,
    replan_enabled: bool | None = None,
    *,
    # LLM step fields
    aim: str = "",
    reasoning_questions: str = "",
    step_context_queries: list[ContextQuery | str] | None = None,
    stage_action: str = "",
    example_reasoning: str = "",
    llm_config: LLMStepConfig | None = None,
    # Non-LLM step config
    config: StepConfig = None,
) -> AnyStepDescription:
    """
    Factory function to create the appropriate step description type.

    This is the preferred way to create steps in new code.

    Args:
        number: Step number in the sequence
        title: Human-readable title
        step_type: Type of step (LLM, TOOL, MCP, MEMORY, TRANSFORM, CONDITIONAL, STRUCTURED_OUTPUT)
        dependencies: List of step numbers this step depends on
        aim: Primary objective (LLM steps only)
        reasoning_questions: Key questions to answer (LLM steps only)
        step_context_queries: Context queries for RAG (LLM steps only)
        stage_action: Specific action (LLM steps only)
        example_reasoning: Example reasoning (LLM steps only)
        llm_config: Per-step LLM configuration override (LLM steps only)
        config: Step configuration (non-LLM steps only)

    Returns:
        The appropriate typed step description instance.
    """
    deps = dependencies or []

    if step_type == StepType.LLM:
        return LLMStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            aim=aim,
            reasoning_questions=reasoning_questions,
            step_context_queries=step_context_queries or [],
            stage_action=stage_action,
            example_reasoning=example_reasoning,
            llm_config=llm_config,
        )
    elif step_type == StepType.TOOL:
        if not isinstance(config, ToolStepConfig):
            raise ValueError("TOOL steps require ToolStepConfig")
        return ToolStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.MCP:
        if not isinstance(config, MCPStepConfig):
            raise ValueError("MCP steps require MCPStepConfig")
        return MCPStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.MEMORY:
        if not isinstance(config, MemoryStepConfig):
            raise ValueError("MEMORY steps require MemoryStepConfig")
        return MemoryStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.TRANSFORM:
        if not isinstance(config, TransformStepConfig):
            raise ValueError("TRANSFORM steps require TransformStepConfig")
        return TransformStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.CONDITIONAL:
        if not isinstance(config, ConditionalStepConfig):
            raise ValueError("CONDITIONAL steps require ConditionalStepConfig")
        return ConditionalStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.STRUCTURED_OUTPUT:
        if not isinstance(config, StructuredOutputStepConfig):
            raise ValueError("STRUCTURED_OUTPUT steps require StructuredOutputStepConfig")
        return StructuredOutputStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    else:
        raise ValueError(f"Unknown step type: {step_type}")
