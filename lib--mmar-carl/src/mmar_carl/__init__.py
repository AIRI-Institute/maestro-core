"""
MMAR CARL (Collaborative Agent Reasoning Library)

A library for building chain-of-thought reasoning systems with DAG-based parallel execution.
Supports multiple step types: LLM, Tool, MCP, Memory, Transform, and Conditional.

Module Structure:
- mmar_carl.models: Package containing all data models (re-exports all for convenience)
  - mmar_carl.models.enums: StepType, MemoryOperation, Language
  - mmar_carl.models.base: SearchStrategy, SelfCriticDecision, SelfCriticEvaluatorBase
  - mmar_carl.models.llm_client_base: LLMClientBase
  - mmar_carl.models.search: SubstringSearchStrategy, VectorSearchStrategy, ContextSearchConfig
  - mmar_carl.models.config: ToolParameter, ToolStepConfig, MCPStepConfig, MemoryStepConfig, etc.
  - mmar_carl.models.steps: StepDescriptionBase, LLMStepDescription, ToolStepDescription, etc.
  - mmar_carl.models.context: ReasoningContext
  - mmar_carl.models.results: StepExecutionResult, ReasoningResult
  - mmar_carl.models.prompts: PromptTemplate

Step Description Classes (New API):
- StepDescriptionBase: Abstract base class for all steps
- LLMStepDescription: LLM reasoning steps
- ToolStepDescription: External tool/function execution
- MCPStepDescription: MCP protocol calls
- MemoryStepDescription: Memory operations
- TransformStepDescription: Data transformations
- ConditionalStepDescription: Conditional branching
- AnyStepDescription: Union type of all step types
- create_step(): Factory function for creating typed steps

Legacy API (Backward Compatible):
- StepDescription: Unified class supporting all step types
"""

from .chain import ChainBuilder, ReasoningChain, ReflectionOptions, create_chain_from_config
from .dataset_evaluator import DatasetEvaluator
from .metrics import MetricBase, MetricOutput
from .executor import DAGExecutor, ExecutionCancelledError
from .tracing import flush as langfuse_flush, is_langfuse_enabled
from .replan import (
    LLMReplanChecker,
    RuleBasedReplanChecker,
)
from .llm import (
    OpenAIClientConfig,
    OpenAICompatibleClient,
    create_openai_client,
)
from .models import (
    # Dataset abstractions
    AbstractDataset,
    CaseEvaluationResult,
    DataCase,
    DataFrameDataset,
    DatasetEvaluationReport,
    SelectionStrategy,
    SimpleDataset,
    ThresholdStrategy,
    TopKWorstStrategy,
    # Enums
    Language,
    MemoryOperation,
    StepType,
    # Abstract base classes
    LLMClientBase,
    SearchStrategy,
    SelfCriticDecision,
    SelfCriticEvaluatorBase,
    # Search & Context
    ContextQuery,
    ContextSearchConfig,
    SubstringSearchStrategy,
    VectorSearchStrategy,
    # Step Configurations
    ConditionalBranch,
    ConditionalStepConfig,
    ExecutionMode,
    LLMStepConfig,
    LLMReplanCheckerConfig,
    MCPServerConfig,
    MCPStepConfig,
    MemoryStepConfig,
    StepConfig,
    ToolParameter,
    ToolStepConfig,
    TransformStepConfig,
    StructuredOutputStepConfig,
    # Step Description Classes (New API)
    AnyStepDescription,
    ConditionalStepDescription,
    LLMStepDescription,
    MCPStepDescription,
    MemoryStepDescription,
    StepDescriptionBase,
    ToolStepDescription,
    TransformStepDescription,
    StructuredOutputStepDescription,
    create_step,
    # Legacy Step Description (Backward Compatible)
    StepDescription,
    # Execution Context & Results
    PromptTemplate,
    RegisteredReplanCheckerConfig,
    ReasoningContext,
    ReasoningResult,
    ReplanAggregationOutcome,
    ReplanCheckerVote,
    ReplanEvent,
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
    StepExecutionResult,
)
from .step_executors import (
    ConditionalStepExecutor,
    LLMStepExecutor,
    MCPStepExecutor,
    MemoryStepExecutor,
    StepExecutorBase,
    ToolStepExecutor,
    TransformStepExecutor,
    StructuredOutputStepExecutor,
    get_executor,
    register_executor,
)
from .logging_utils import (
    get_logger,
    set_log_level,
    log_chain_start,
    log_chain_complete,
    log_step_start,
    log_step_complete,
    log_batch_start,
    log_error,
    log_warning,
    log_debug,
    log_info,
)

__version__ = "0.2.0"
__all__ = [
    # Core Chain Classes
    "ReasoningChain",
    "ChainBuilder",
    "DAGExecutor",
    "create_chain_from_config",
    # Metrics
    "MetricBase",
    "MetricOutput",
    # Reflection
    "ReflectionOptions",
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
    "DatasetEvaluator",
    # Exceptions
    "ExecutionCancelledError",
    # Enums
    "Language",
    "MemoryOperation",
    "StepType",
    # Abstract Base Classes
    "LLMClientBase",
    "SearchStrategy",
    "SelfCriticDecision",
    "SelfCriticEvaluatorBase",
    # Search & Context
    "ContextQuery",
    "ContextSearchConfig",
    "SubstringSearchStrategy",
    "VectorSearchStrategy",
    # Step Configurations
    "ConditionalBranch",
    "ConditionalStepConfig",
    "ExecutionMode",
    "LLMStepConfig",
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
    "MCPServerConfig",
    "MCPStepConfig",
    "MemoryStepConfig",
    "StepConfig",
    "ToolParameter",
    "ToolStepConfig",
    "TransformStepConfig",
    "StructuredOutputStepConfig",
    # Step Description Classes (New API)
    "AnyStepDescription",
    "ConditionalStepDescription",
    "LLMStepDescription",
    "MCPStepDescription",
    "MemoryStepDescription",
    "StepDescriptionBase",
    "ToolStepDescription",
    "TransformStepDescription",
    "StructuredOutputStepDescription",
    "create_step",
    # Legacy Step Description (Backward Compatible)
    "StepDescription",
    # Execution Context & Results
    "PromptTemplate",
    "ReasoningContext",
    "ReasoningResult",
    "ReplanCheckerVote",
    "ReplanAggregationOutcome",
    "ReplanEvent",
    "StepExecutionResult",
    # Tracing
    "langfuse_flush",
    "is_langfuse_enabled",
    # LLM
    "create_openai_client",
    "OpenAIClientConfig",
    "OpenAICompatibleClient",
    # RE-PLAN runtime checkers
    "RuleBasedReplanChecker",
    "LLMReplanChecker",
    # Step Executors
    "ConditionalStepExecutor",
    "LLMStepExecutor",
    "MCPStepExecutor",
    "MemoryStepExecutor",
    "StepExecutorBase",
    "ToolStepExecutor",
    "TransformStepExecutor",
    "StructuredOutputStepExecutor",
    "get_executor",
    "register_executor",
    # Logging
    "get_logger",
    "set_log_level",
    "log_chain_start",
    "log_chain_complete",
    "log_step_start",
    "log_step_complete",
    "log_batch_start",
    "log_error",
    "log_warning",
    "log_debug",
    "log_info",
]
