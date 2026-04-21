"""Models and interfaces for chain-level RE-PLAN policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

from .enums import StepType


class ReplanAction(StrEnum):
    """Action selected by RE-PLAN policy after a checker evaluation."""

    CONTINUE = "continue"
    RETRY_CURRENT_STEP = "retry_current_step"
    REPLAN_FROM_CHECKPOINT = "replan_from_checkpoint"
    RESTART_CHAIN = "restart_chain"
    FAIL = "fail"


class ReplanTargetType(StrEnum):
    """Supported rollback targets for RE-PLAN actions."""

    CHAIN_START = "chain_start"
    CURRENT_STEP = "current_step"
    NEAREST_CHECKPOINT = "nearest_checkpoint"
    NAMED_CHECKPOINT = "named_checkpoint"
    STEP_NUMBER = "step_number"


class ReplanRollbackTarget(BaseModel):
    """Specific rollback target for a RE-PLAN action."""

    target_type: ReplanTargetType = Field(..., description="Rollback target type")
    checkpoint_name: str | None = Field(default=None, description="Checkpoint name for NAMED_CHECKPOINT")
    step_number: int | None = Field(default=None, description="Step number for STEP_NUMBER")
    reason: str = Field(default="", description="Optional note about target choice")

    @model_validator(mode="after")
    def _validate_target_fields(self) -> "ReplanRollbackTarget":
        if self.target_type == ReplanTargetType.NAMED_CHECKPOINT and not self.checkpoint_name:
            raise ValueError("checkpoint_name is required when target_type='named_checkpoint'")
        if self.target_type == ReplanTargetType.STEP_NUMBER and self.step_number is None:
            raise ValueError("step_number is required when target_type='step_number'")
        return self

    def to_key(self) -> str:
        """Return a stable key for budget/accounting checks."""
        if self.target_type == ReplanTargetType.NAMED_CHECKPOINT:
            return f"named:{self.checkpoint_name}"
        if self.target_type == ReplanTargetType.STEP_NUMBER:
            return f"step:{self.step_number}"
        return str(self.target_type.value)


class ReplanVerdict(BaseModel):
    """Structured checker verdict for RE-PLAN decisions."""

    action: ReplanAction = Field(default=ReplanAction.CONTINUE, description="Requested chain-level action")
    reason: str = Field(default="", description="Human-readable reason for this verdict")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Optional confidence score")
    suggested_target: ReplanRollbackTarget | None = Field(
        default=None,
        description="Optional rollback target for replan/restart actions",
    )
    regeneration_hints: list[str] = Field(
        default_factory=list,
        description="Hints passed to regenerated executions",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Checker-specific metadata")

    def is_replan(self) -> bool:
        """Return True when this verdict requests action beyond normal continuation."""
        return self.action != ReplanAction.CONTINUE


class ReplanAggregationStrategy(StrEnum):
    """How multiple checker verdicts are aggregated."""

    ANY = "any"
    ALL = "all"
    K_OF_N = "k_of_n"
    MANDATORY_PLUS_K_OF_REST = "mandatory_plus_k_of_rest"


class ReplanAggregationConfig(BaseModel):
    """Configuration for combining multiple checker verdicts."""

    strategy: ReplanAggregationStrategy = Field(
        default=ReplanAggregationStrategy.ANY,
        description="Aggregation strategy for checker votes",
    )
    k: int = Field(default=1, ge=1, description="K threshold for K_OF_N / MANDATORY_PLUS_K_OF_REST")
    mandatory_checkers: list[str] = Field(
        default_factory=list,
        description="Checker names that must request replan in MANDATORY_PLUS_K_OF_REST",
    )
    action_selection: Literal["priority", "highest_confidence"] = Field(
        default="priority",
        description="How to choose final action when multiple non-continue votes exist",
    )


class ReplanTriggerConfig(BaseModel):
    """When RE-PLAN checks should run."""

    evaluate_after_step: bool = Field(default=True, description="Evaluate after each completed step")
    evaluate_after_failure: bool = Field(default=True, description="Evaluate after failed steps")
    checkpoint_only: bool = Field(
        default=False,
        description="If true, evaluate only for steps marked as checkpoints",
    )
    step_numbers: list[int] = Field(
        default_factory=list,
        description="Optional allow-list of step numbers for trigger evaluation",
    )
    step_types: list[StepType] = Field(
        default_factory=list,
        description="Optional allow-list of step types for trigger evaluation",
    )


class ReplanBudgetConfig(BaseModel):
    """Safeguards to prevent infinite replanning loops."""

    max_replans_per_chain: int = Field(default=3, ge=0, description="Maximum total replan actions per chain run")
    max_replans_per_step: int = Field(default=2, ge=0, description="Maximum replan actions attributable to one step")
    max_visits_per_checkpoint: int = Field(
        default=3,
        ge=0,
        description="Maximum times the same rollback target may be revisited",
    )
    max_same_rollback_target_repeats: int = Field(
        default=2,
        ge=0,
        description="Maximum consecutive replans to the same rollback target",
    )
    fail_on_budget_exhaustion: bool = Field(
        default=True,
        description="If true, execution fails when any budget is exhausted",
    )


class RuleBasedReplanCheckerConfig(BaseModel):
    """Config for a deterministic rule-based RE-PLAN checker."""

    type: Literal["rule_based"] = "rule_based"
    name: str = Field(default="rule_based")
    trigger_on_failed_step: bool = Field(default=True)
    error_substrings: list[str] = Field(default_factory=list)
    result_substrings: list[str] = Field(default_factory=list)
    action_on_failure: ReplanAction = Field(default=ReplanAction.RETRY_CURRENT_STEP)
    action_on_match: ReplanAction = Field(default=ReplanAction.REPLAN_FROM_CHECKPOINT)
    rollback_target_on_failure: ReplanRollbackTarget | None = Field(default=None)
    rollback_target_on_match: ReplanRollbackTarget | None = Field(default=None)
    feedback_on_failure: list[str] = Field(default_factory=list)
    feedback_on_match: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


_DEFAULT_LLM_REPLAN_PROMPT = """You are a chain-level replanning controller.
You must decide whether execution should continue as-is or trigger replanning.
Return ONLY a valid JSON object matching this schema:
${schema_json}

Current step:
- number: ${step_number}
- title: ${step_title}
- type: ${step_type}
- success: ${step_success}
- result: ${step_result}
- error: ${step_error}

Recent history:
${history}

Recent errors:
${recent_errors}

Available checkpoints:
${checkpoint_names}

Budget snapshot:
${budget_snapshot}

When uncertain, prefer "continue".
"""


class LLMReplanCheckerConfig(BaseModel):
    """Config for an LLM-based structured RE-PLAN checker."""

    type: Literal["llm"] = "llm"
    name: str = Field(default="llm_replan")
    prompt_template: str = Field(default=_DEFAULT_LLM_REPLAN_PROMPT)
    model: str | None = Field(default=None, description="Optional per-checker model override")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    retries: int = Field(default=1, ge=1)
    history_entries: int = Field(default=6, ge=0)
    error_entries: int = Field(default=4, ge=0)
    parse_error_action: ReplanAction = Field(
        default=ReplanAction.FAIL,
        description="Action returned when model output cannot be parsed into ReplanVerdict",
    )


class RegisteredReplanCheckerConfig(BaseModel):
    """Reference to a checker registered in ReasoningContext."""

    type: Literal["registered"] = "registered"
    name: str = Field(..., description="Name from context.register_replan_checker(name, checker)")


ReplanCheckerSpec = Union[
    RuleBasedReplanCheckerConfig,
    LLMReplanCheckerConfig,
    RegisteredReplanCheckerConfig,
]


class ReplanPolicy(BaseModel):
    """Chain-level RE-PLAN policy configuration."""

    enabled: bool = Field(default=True, description="Enable chain-level RE-PLAN policy")
    checkers: list[ReplanCheckerSpec] = Field(default_factory=list, description="Configured checker specifications")
    aggregation: ReplanAggregationConfig = Field(default_factory=ReplanAggregationConfig)
    trigger: ReplanTriggerConfig = Field(default_factory=ReplanTriggerConfig)
    budgets: ReplanBudgetConfig = Field(default_factory=ReplanBudgetConfig)
    default_checkpoint_target: ReplanRollbackTarget = Field(
        default_factory=lambda: ReplanRollbackTarget(target_type=ReplanTargetType.NEAREST_CHECKPOINT),
    )


class ReplanCheckerInput(BaseModel):
    """Structured input passed to RE-PLAN checkers."""

    step_number: int
    step_title: str
    step_type: StepType
    step_success: bool
    step_result: str = ""
    step_result_data: Any = None
    step_error: str | None = None
    history: list[str] = Field(default_factory=list)
    recent_errors: list[str] = Field(default_factory=list)
    checkpoint_names: list[str] = Field(default_factory=list)
    checkpoint_steps: list[int] = Field(default_factory=list)
    budget_snapshot: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplanCheckerBase(ABC):
    """Abstract interface for custom RE-PLAN checkers."""

    @abstractmethod
    async def evaluate(self, checker_input: ReplanCheckerInput, context: Any) -> ReplanVerdict:
        """Evaluate current execution state and return a structured RE-PLAN verdict."""
        raise NotImplementedError
