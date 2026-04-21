"""
Result classes for CARL reasoning system.
"""

from typing import Any

from pydantic import BaseModel, Field

from .enums import StepType
from .replan import ReplanAction, ReplanAggregationStrategy, ReplanRollbackTarget


class StepExecutionResult(BaseModel):
    """
    Result of executing a single reasoning step.
    """

    step_number: int = Field(..., description="Number of the executed step")
    step_title: str = Field(..., description="Title of the executed step")
    step_type: StepType = Field(default=StepType.LLM, description="Type of step that was executed")
    result: str = Field(..., description="Result content (string representation)")
    result_data: Any = Field(default=None, description="Structured result data (for non-LLM steps)")
    success: bool = Field(..., description="Whether execution succeeded")
    error_message: str | None = Field(default=None, description="Error message if execution failed")
    error_traceback: str | None = Field(default=None, description="Full traceback if execution failed")
    execution_time: float | None = Field(default=None, description="Time taken for execution in seconds")
    updated_history: list[str] = Field(default_factory=list, description="History after this step's execution")
    token_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage for this step: {'prompt': X, 'completion': Y, 'total': Z}"
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Metric scores for this step: {metric_name: score}",
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the step result to a dictionary.

        Useful for JSON serialization and logging.
        """
        return {
            "step_number": self.step_number,
            "step_title": self.step_title,
            "step_type": str(self.step_type),
            "success": self.success,
            "result": self.result[:1000] if self.result else None,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "execution_time": self.execution_time,
            "token_usage": self.token_usage,
            "metrics": self.metrics,
        }


class ReplanCheckerVote(BaseModel):
    """Single checker verdict recorded for a RE-PLAN evaluation."""

    checker_name: str
    action: ReplanAction
    reason: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    suggested_target: ReplanRollbackTarget | None = None
    regeneration_hints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplanAggregationOutcome(BaseModel):
    """Aggregated decision outcome across all checker votes."""

    strategy: ReplanAggregationStrategy
    triggered: bool
    trigger_count: int
    total_count: int
    mandatory_satisfied: bool = True
    selected_checker: str | None = None
    selected_action: ReplanAction = ReplanAction.CONTINUE


class ReplanEvent(BaseModel):
    """A recorded RE-PLAN evaluation/action event."""

    sequence: int
    step_number: int
    step_title: str
    checker_votes: list[ReplanCheckerVote] = Field(default_factory=list)
    aggregation: ReplanAggregationOutcome
    final_action: ReplanAction = ReplanAction.CONTINUE
    rollback_target: ReplanRollbackTarget | None = None
    feedback_passed: list[str] = Field(default_factory=list)
    triggering_checkers: list[str] = Field(default_factory=list)
    budget_usage: dict[str, Any] = Field(default_factory=dict)
    budget_exhausted: bool = False
    note: str = ""


class ReasoningResult(BaseModel):
    """
    Final result of executing a complete reasoning chain.
    """

    success: bool = Field(..., description="Whether overall execution succeeded")
    history: list[str] = Field(..., description="Complete reasoning history")
    step_results: list[StepExecutionResult] = Field(..., description="Results from each step")
    total_execution_time: float | None = Field(default=None, description="Total execution time in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")
    replan_events: list[ReplanEvent] = Field(default_factory=list, description="Recorded RE-PLAN events")
    token_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Total token usage across all steps: {'prompt': X, 'completion': Y, 'total': Z}"
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Chain-level metric scores: {metric_name: score}",
    )

    @property
    def error(self) -> str | None:
        """
        Get the error message from the first failed step.

        Returns:
            Error message string if any step failed, None otherwise
        """
        for step in self.step_results:
            if not step.success and step.error_message:
                return step.error_message
        return None

    def get_full_output(self) -> str:
        """Get the full reasoning output as a single string."""
        return "\n".join(self.history)

    def get_final_output(self) -> str:
        """Get the final reasoning output as a single string without step headers."""
        if not self.history:
            return ""
        last_entry = self.history[-1]
        # Check if it's a step result with header (Russian or English)
        if last_entry.startswith("Шаг ") and "\nРезультат: " in last_entry:
            # Extract content after "Результат: " for Russian steps
            return last_entry.split("\nРезультат: ", 1)[1].strip()
        elif last_entry.startswith("Step ") and "\nResult: " in last_entry:
            # Extract content after "Result: " for English steps
            return last_entry.split("\nResult: ", 1)[1].strip()
        else:
            # Return as-is if it doesn't match expected format
            return last_entry.strip()

    def get_successful_steps(self) -> list[StepExecutionResult]:
        """Get all successfully executed steps."""
        return [step for step in self.step_results if step.success]

    def get_failed_steps(self) -> list[StepExecutionResult]:
        """Get all failed steps."""
        return [step for step in self.step_results if not step.success]

    def get_step_result(self, step_number: int) -> StepExecutionResult | None:
        """
        Get the result for a specific step by its number.

        Args:
            step_number: The step number to look up

        Returns:
            The StepExecutionResult for that step, or None if not found
        """
        for result in self.step_results:
            if result.step_number == step_number:
                return result
        return None

    def get_total_tokens(self) -> dict[str, int]:
        """
        Calculate total token usage across all steps.

        Returns:
            Dict with 'prompt', 'completion', and 'total' token counts
        """
        total_prompt = 0
        total_completion = 0

        for step in self.step_results:
            if step.token_usage:
                total_prompt += step.token_usage.get("prompt", 0)
                total_completion += step.token_usage.get("completion", 0)

        return {
            "prompt": total_prompt,
            "completion": total_completion,
            "total": total_prompt + total_completion,
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the reasoning result to a dictionary.

        Useful for JSON serialization and logging.
        """
        return {
            "success": self.success,
            "total_execution_time": self.total_execution_time,
            "total_steps": len(self.step_results),
            "successful_steps": len(self.get_successful_steps()),
            "failed_steps": len(self.get_failed_steps()),
            "token_usage": self.token_usage or self.get_total_tokens(),
            "metrics": self.metrics,
            "step_results": [r.to_dict() for r in self.step_results],
            "replan_events": [event.model_dump(mode="json") for event in self.replan_events],
            "metadata": self.metadata,
        }
