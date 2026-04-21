"""
Core data models for CARL reasoning system.

This module re-exports all models for backward compatibility.
For new code, prefer importing from the specific submodules:
- mmar_carl.models.enums: StepType, MemoryOperation, Language
- mmar_carl.models.base: SearchStrategy, SelfCriticDecision, SelfCriticEvaluatorBase
- mmar_carl.models.llm_client_base: SearchStrategy, SelfCriticDecision, SelfCriticEvaluatorBase
- mmar_carl.models.search: SubstringSearchStrategy, VectorSearchStrategy, ContextSearchConfig
- mmar_carl.models.config: ToolParameter, ToolStepConfig, MCPStepConfig, etc.
- mmar_carl.models.steps: StepDescriptionBase, LLMStepDescription, etc.
- mmar_carl.models.context: ReasoningContext
- mmar_carl.models.results: StepExecutionResult, ReasoningResult
- mmar_carl.models.prompts: PromptTemplate
"""

# Re-export all public symbols for backward compatibility
# flake8: noqa: F401

# Enums
from .enums import Language, MemoryOperation, StepType

# Abstract base classes
from .base import SearchStrategy, SelfCriticDecision, SelfCriticEvaluatorBase
from .llm_client_base import LLMClientBase

# Search strategies
from .search import ContextSearchConfig, SubstringSearchStrategy, VectorSearchStrategy

# Step configurations
from .config import (
    ConditionalBranch,
    ConditionalStepConfig,
    ContextQuery,
    ExecutionMode,
    LLMStepConfig,
    MCPServerConfig,
    MCPStepConfig,
    MemoryStepConfig,
    StepConfig,
    ToolParameter,
    ToolStepConfig,
    TransformStepConfig,
    StructuredOutputStepConfig,
)

# RE-PLAN models
from .replan import (
    LLMReplanCheckerConfig,
    RegisteredReplanCheckerConfig,
    ReplanAction,
    ReplanAggregationConfig,
    ReplanAggregationStrategy,
    ReplanBudgetConfig,
    ReplanCheckerBase,
    ReplanCheckerInput,
    ReplanCheckerSpec,
    ReplanPolicy,
    ReplanRollbackTarget,
    ReplanTargetType,
    ReplanTriggerConfig,
    ReplanVerdict,
    RuleBasedReplanCheckerConfig,
)

# Step descriptions
from .steps import (
    AnyStepDescription,
    ConditionalStepDescription,
    LLMStepDescription,
    MCPStepDescription,
    MemoryStepDescription,
    StepDescription,
    StepDescriptionBase,
    ToolStepDescription,
    TransformStepDescription,
    StructuredOutputStepDescription,
    create_step,
)

# Context
from .context import ReasoningContext

# Results
from .results import (
    ReasoningResult,
    ReplanAggregationOutcome,
    ReplanCheckerVote,
    ReplanEvent,
    StepExecutionResult,
)

# Prompts
from .prompts import PromptTemplate

# Dataset abstractions
from .dataset import (
    AbstractDataset,
    CaseEvaluationResult,
    DataCase,
    DataFrameDataset,
    DatasetEvaluationReport,
    SelectionStrategy,
    SimpleDataset,
    ThresholdStrategy,
    TopKWorstStrategy,
)

# Define __all__ for explicit public API
__all__ = [
    # Enums
    "StepType",
    "MemoryOperation",
    "Language",
    # Abstract base classes
    "LLMClientBase",
    "SearchStrategy",
    "SelfCriticDecision",
    "SelfCriticEvaluatorBase",
    # Search strategies
    "SubstringSearchStrategy",
    "VectorSearchStrategy",
    "ContextSearchConfig",
    # Step configurations
    "ToolParameter",
    "ToolStepConfig",
    "MCPServerConfig",
    "MCPStepConfig",
    "MemoryStepConfig",
    "TransformStepConfig",
    "StructuredOutputStepConfig",
    "ConditionalBranch",
    "ConditionalStepConfig",
    "StepConfig",
    "ContextQuery",
    "ExecutionMode",
    "LLMStepConfig",
    # RE-PLAN
    "ReplanAction",
    "ReplanTargetType",
    "ReplanRollbackTarget",
    "ReplanVerdict",
    "ReplanAggregationStrategy",
    "ReplanAggregationConfig",
    "ReplanTriggerConfig",
    "ReplanBudgetConfig",
    "RuleBasedReplanCheckerConfig",
    "LLMReplanCheckerConfig",
    "RegisteredReplanCheckerConfig",
    "ReplanCheckerSpec",
    "ReplanPolicy",
    "ReplanCheckerInput",
    "ReplanCheckerBase",
    # Step descriptions
    "StepDescriptionBase",
    "LLMStepDescription",
    "ToolStepDescription",
    "MCPStepDescription",
    "MemoryStepDescription",
    "TransformStepDescription",
    "StructuredOutputStepDescription",
    "ConditionalStepDescription",
    "AnyStepDescription",
    "StepDescription",
    "create_step",
    # Context
    "ReasoningContext",
    # Results
    "StepExecutionResult",
    "ReplanCheckerVote",
    "ReplanAggregationOutcome",
    "ReplanEvent",
    "ReasoningResult",
    # Prompts
    "PromptTemplate",
    # Dataset abstractions
    "DataCase",
    "AbstractDataset",
    "SimpleDataset",
    "DataFrameDataset",
    "ThresholdStrategy",
    "TopKWorstStrategy",
    "SelectionStrategy",
    "CaseEvaluationResult",
    "DatasetEvaluationReport",
]
