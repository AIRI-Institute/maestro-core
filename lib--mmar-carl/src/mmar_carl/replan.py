"""Runtime RE-PLAN checker implementations and aggregation helpers."""

from __future__ import annotations

import json
import string
from typing import Any

from pydantic import BaseModel, ValidationError

from .models import LLMStepConfig, ReasoningContext
from .models.replan import (
    LLMReplanCheckerConfig,
    RegisteredReplanCheckerConfig,
    ReplanAction,
    ReplanAggregationConfig,
    ReplanAggregationStrategy,
    ReplanCheckerBase,
    ReplanCheckerInput,
    ReplanCheckerSpec,
    ReplanVerdict,
    RuleBasedReplanCheckerConfig,
)


class AggregatedReplanDecision(BaseModel):
    """Aggregated RE-PLAN decision from checker votes."""

    triggered: bool
    selected_checker: str | None
    selected_verdict: ReplanVerdict
    trigger_count: int
    total_count: int
    mandatory_satisfied: bool
    triggering_checkers: list[str]


class CheckerVote(BaseModel):
    """Checker vote used by the aggregation logic."""

    checker_name: str
    verdict: ReplanVerdict


_ACTION_PRIORITY: dict[ReplanAction, int] = {
    ReplanAction.CONTINUE: 0,
    ReplanAction.RETRY_CURRENT_STEP: 1,
    ReplanAction.REPLAN_FROM_CHECKPOINT: 2,
    ReplanAction.RESTART_CHAIN: 3,
    ReplanAction.FAIL: 4,
}


class RuleBasedReplanChecker(ReplanCheckerBase):
    """Deterministic checker using simple result/error substring rules."""

    def __init__(self, config: RuleBasedReplanCheckerConfig):
        self.config = config

    async def evaluate(self, checker_input: ReplanCheckerInput, context: Any) -> ReplanVerdict:
        _ = context
        result_text = (checker_input.step_result or "").lower()
        error_text = (checker_input.step_error or "").lower()

        if (not checker_input.step_success) and self.config.trigger_on_failed_step:
            reason = (
                f"Step {checker_input.step_number} failed. "
                f"Rule-based checker '{self.config.name}' requested {self.config.action_on_failure.value}."
            )
            return ReplanVerdict(
                action=self.config.action_on_failure,
                reason=reason,
                confidence=self.config.confidence,
                suggested_target=self.config.rollback_target_on_failure,
                regeneration_hints=self.config.feedback_on_failure.copy(),
            )

        if self.config.error_substrings and any(substr.lower() in error_text for substr in self.config.error_substrings):
            reason = (
                f"Error text matched rule in checker '{self.config.name}'. "
                f"Requested {self.config.action_on_match.value}."
            )
            return ReplanVerdict(
                action=self.config.action_on_match,
                reason=reason,
                confidence=self.config.confidence,
                suggested_target=self.config.rollback_target_on_match,
                regeneration_hints=self.config.feedback_on_match.copy(),
            )

        if self.config.result_substrings and any(substr.lower() in result_text for substr in self.config.result_substrings):
            reason = (
                f"Result text matched rule in checker '{self.config.name}'. "
                f"Requested {self.config.action_on_match.value}."
            )
            return ReplanVerdict(
                action=self.config.action_on_match,
                reason=reason,
                confidence=self.config.confidence,
                suggested_target=self.config.rollback_target_on_match,
                regeneration_hints=self.config.feedback_on_match.copy(),
            )

        return ReplanVerdict(action=ReplanAction.CONTINUE, reason=f"Checker '{self.config.name}' did not match any rule.")


class LLMReplanChecker(ReplanCheckerBase):
    """LLM-based checker that returns strict structured RE-PLAN verdicts."""

    def __init__(self, config: LLMReplanCheckerConfig):
        self.config = config

    def _build_prompt(self, checker_input: ReplanCheckerInput) -> str:
        history_slice = checker_input.history[-self.config.history_entries :] if self.config.history_entries else []
        errors_slice = checker_input.recent_errors[-self.config.error_entries :] if self.config.error_entries else []

        # Use string.Template to avoid conflicts with JSON braces in the schema
        render_map = {
            "schema_json": json.dumps(ReplanVerdict.model_json_schema(), ensure_ascii=False, indent=2),
            "step_number": str(checker_input.step_number),
            "step_title": checker_input.step_title,
            "step_type": str(checker_input.step_type),
            "step_success": str(checker_input.step_success),
            "step_result": checker_input.step_result or "",
            "step_error": checker_input.step_error or "",
            "history": "\n".join(history_slice) if history_slice else "(empty)",
            "recent_errors": "\n".join(errors_slice) if errors_slice else "(none)",
            "checkpoint_names": ", ".join(checker_input.checkpoint_names) if checker_input.checkpoint_names else "(none)",
            "budget_snapshot": json.dumps(checker_input.budget_snapshot, ensure_ascii=False, indent=2),
        }
        template = string.Template(self.config.prompt_template)
        return template.safe_substitute(render_map)

    async def evaluate(self, checker_input: ReplanCheckerInput, context: ReasoningContext) -> ReplanVerdict:
        override: LLMStepConfig | None = None
        if any(
            value is not None
            for value in (self.config.model, self.config.temperature, self.config.max_tokens)
        ):
            override = LLMStepConfig(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        llm_client = context.get_llm_client_for_step(override)
        prompt = self._build_prompt(checker_input)

        raw_response = await llm_client.get_response_with_retries(prompt, retries=self.config.retries)
        normalized = (raw_response or "").strip()

        try:
            verdict = ReplanVerdict.model_validate_json(normalized)
        except ValidationError as exc:
            return ReplanVerdict(
                action=self.config.parse_error_action,
                reason=(
                    f"LLM checker '{self.config.name}' returned invalid structured output: {exc.errors()[0]['msg']}"
                ),
                confidence=0.0,
                metadata={"checker": self.config.name, "parse_error": str(exc), "raw_response": normalized[:1000]},
            )
        except Exception as exc:
            return ReplanVerdict(
                action=self.config.parse_error_action,
                reason=f"LLM checker '{self.config.name}' parsing failed: {exc}",
                confidence=0.0,
                metadata={"checker": self.config.name, "parse_error": str(exc), "raw_response": normalized[:1000]},
            )

        return verdict


def create_checker_from_spec(spec: ReplanCheckerSpec, context: ReasoningContext) -> ReplanCheckerBase:
    """Instantiate checker runtime implementation from policy specification."""
    if isinstance(spec, RuleBasedReplanCheckerConfig):
        return RuleBasedReplanChecker(spec)

    if isinstance(spec, LLMReplanCheckerConfig):
        return LLMReplanChecker(spec)

    if isinstance(spec, RegisteredReplanCheckerConfig):
        checker = context.get_replan_checker(spec.name)
        if checker is None:
            raise ValueError(
                f"RE-PLAN checker '{spec.name}' is not registered. "
                f"Available: {context.list_replan_checkers()}"
            )
        return checker

    raise TypeError(f"Unsupported RE-PLAN checker specification: {type(spec).__name__}")


def _pick_selected_vote(votes: list[CheckerVote], config: ReplanAggregationConfig) -> CheckerVote | None:
    if not votes:
        return None

    if config.action_selection == "highest_confidence":
        return max(
            votes,
            key=lambda vote: ((vote.verdict.confidence or 0.0), _ACTION_PRIORITY.get(vote.verdict.action, -1)),
        )

    return max(
        votes,
        key=lambda vote: (_ACTION_PRIORITY.get(vote.verdict.action, -1), (vote.verdict.confidence or 0.0)),
    )


def aggregate_replan_votes(votes: list[CheckerVote], config: ReplanAggregationConfig) -> AggregatedReplanDecision:
    """Aggregate checker verdicts according to policy configuration."""
    total = len(votes)
    trigger_votes = [vote for vote in votes if vote.verdict.is_replan()]
    trigger_count = len(trigger_votes)

    fail_votes = [vote for vote in trigger_votes if vote.verdict.action == ReplanAction.FAIL]
    if fail_votes:
        selected = _pick_selected_vote(fail_votes, config) or fail_votes[0]
        return AggregatedReplanDecision(
            triggered=True,
            selected_checker=selected.checker_name,
            selected_verdict=selected.verdict,
            trigger_count=trigger_count,
            total_count=total,
            mandatory_satisfied=True,
            triggering_checkers=[vote.checker_name for vote in trigger_votes],
        )

    mandatory_satisfied = True
    triggered = False

    if total == 0:
        selected_verdict = ReplanVerdict(action=ReplanAction.CONTINUE, reason="No RE-PLAN checkers configured.")
        return AggregatedReplanDecision(
            triggered=False,
            selected_checker=None,
            selected_verdict=selected_verdict,
            trigger_count=0,
            total_count=0,
            mandatory_satisfied=True,
            triggering_checkers=[],
        )

    if config.strategy == ReplanAggregationStrategy.ANY:
        triggered = trigger_count >= 1

    elif config.strategy == ReplanAggregationStrategy.ALL:
        triggered = trigger_count == total

    elif config.strategy == ReplanAggregationStrategy.K_OF_N:
        threshold = min(max(config.k, 1), total)
        triggered = trigger_count >= threshold

    elif config.strategy == ReplanAggregationStrategy.MANDATORY_PLUS_K_OF_REST:
        mandatory = {name for name in config.mandatory_checkers if name}
        vote_map = {vote.checker_name: vote for vote in votes}
        mandatory_satisfied = all(
            name in vote_map and vote_map[name].verdict.is_replan() for name in mandatory
        )
        non_mandatory_trigger_count = len(
            [vote for vote in trigger_votes if vote.checker_name not in mandatory]
        )
        threshold = max(config.k, 0)
        triggered = mandatory_satisfied and non_mandatory_trigger_count >= threshold

    selected_vote = _pick_selected_vote(trigger_votes, config) if triggered else None
    if selected_vote is None:
        selected_verdict = ReplanVerdict(action=ReplanAction.CONTINUE, reason="Aggregation policy voted to continue.")
    else:
        selected_verdict = selected_vote.verdict.model_copy(deep=True)

    return AggregatedReplanDecision(
        triggered=triggered,
        selected_checker=selected_vote.checker_name if selected_vote else None,
        selected_verdict=selected_verdict,
        trigger_count=trigger_count,
        total_count=total,
        mandatory_satisfied=mandatory_satisfied,
        triggering_checkers=[vote.checker_name for vote in trigger_votes],
    )
