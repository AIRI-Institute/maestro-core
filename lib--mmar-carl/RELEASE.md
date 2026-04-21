# Release Notes

Detailed release notes for CARL (Multi-step Agentic Reasoning with LLMs) library.

## Version 0.2.0 - 2026-04-15

### 🔧 Breaking Changes

#### LLM Client Refactoring

**Moved LLMClientBase to separate module**
- `LLMClientBase` moved from `mmar_carl.models.base` to `mmar_carl.models.llm_client_base`
- Update imports if you're directly importing this class:
  ```python
  # Old (deprecated)
  from mmar_carl.models.base import LLMClientBase

  # New (correct)
  from mmar_carl.models.llm_client_base import LLMClientBase
  ```

**Removed legacy mmar-llm-mapi integration**
- Completely removed integration with the deprecated `mmar-llm-mapi` library
- Removed `create_llm_client()` function
- Use `create_openai_client()` for OpenAI-compatible APIs instead

**Rationale:** Simplifies the architecture and removes dependencies on deprecated libraries.

---

### ✨ New Features

#### Comprehensive Test Suite

Added 74 new comprehensive tests (4,237 lines of test code):

**New Test Files:**
- `tests/test_conditional_steps.py` - 20 tests for conditional branching patterns
  - Built-in patterns: contains, equals, startswith, endswith, matches, empty, nonempty
  - Complex expressions with simpleeval
  - Multi-branch routing with default steps
  - Serialization and ChainBuilder integration

- `tests/test_advanced_tool_steps.py` - 13 tests for tool integration
  - Multi-step tool chains with parallel execution
  - Input mapping with $metadata, $history, $outer_context references
  - Tool error handling and parameter mapping
  - Complex data flow between tool steps

- `tests/test_llm_council.py` - 10 tests for multi-model voting patterns
  - Parallel council member execution
  - Per-step model overrides
  - Vote aggregation (unanimous, majority, split)
  - Council synthesis and complex scenarios

- `tests/test_structured_output_advanced.py` - 17 tests for JSON schema validation
  - Pydantic model validation
  - JSON schema validation (nested, complex)
  - Error recovery from invalid JSON
  - Strict vs lenient parsing modes

- `tests/test_execution_modes_advanced.py` - 14 tests for FAST/SELF_CRITIC modes
  - FAST mode single-pass behavior
  - SELF_CRITIC with custom evaluators
  - Evaluator chains and revision limits
  - Mixed execution modes and performance characteristics

**Enhanced Mock Infrastructure:**
- `tests/mocks.py` - 12 specialized mock clients for testing
  - MockLLMClient - Basic mock for general testing
  - ConditionalMockClient - Pattern-based responses
  - CouncilMockClient - Multi-model council simulation
  - ToolTrackingMockClient - Tracks tool execution flow
  - StructuredOutputMockClient - JSON response simulation
  - ReplanScenarioMockClient - RE-PLAN scenario simulation
  - ExecutionModeMockClient - FAST vs SELF_CRITIC mode testing
  - ChainBuilderMockClient - Chain building validation
  - And more...

**Test Coverage:** 266 tests total (192 existing + 74 new) - 100% pass rate

#### Examples Runner

**New `examples/runner.py`** - Universal runner for all examples

```bash
# Run any example with automatic setup
python examples/runner.py --example basic_chain_example

# List available examples
python examples/runner.py --list

# Run with custom parameters
python examples/runner.py --example llm_council_example --model claude-3-5-sonnet
```

Benefits:
- Simplified example execution without manual setup
- Better testing and demonstration capabilities
- Consistent environment across all examples
- Easy to add new examples

---

### 🐛 Bug Fixes

#### BUG-001 (HIGH): Conditional Steps Execute All Branches

**Problem:** Conditional steps were executing ALL possible branch target steps instead of only the matched branch.

**Fix:** DAG executor now respects conditional routing decisions
- Added `_skip_conditional_branches()` helper method
- Added `_is_reachable_from_target()` helper method
- Only executes steps that are reachable from the matched branch

**Before:**
```python
# Conditional step matches condition "int(value) >= 70"
# Routes to step 3 (High Score)
# BUG: Executes steps 1, 3, 4, 2 (all branches)
assert len(result.step_results) == 4  # Wrong!
```

**After:**
```python
# Conditional step matches condition "int(value) >= 70"
# Routes to step 3 (High Score)
# CORRECT: Executes only steps 1, 2, 3 (matched branch)
assert len(result.step_results) == 3  # Correct!
```

**Files Modified:** `src/mmar_carl/executor.py`

---

#### BUG-002 (MEDIUM): Step Type Field Missing from Serialization

**Problem:** The `step_type` field was not included when step descriptions were serialized using `model_dump()`.

**Fix:** Added `model_dump()` override to all 7 step description classes
- `LLMStepDescription`
- `ToolStepDescription`
- `MCPStepDescription`
- `MemoryStepDescription`
- `TransformStepDescription`
- `ConditionalStepDescription`
- `StructuredOutputStepDescription`

**Before:**
```python
step_dict = cond_step.model_dump()
assert "step_type" not in step_dict  # Missing!
```

**After:**
```python
step_dict = cond_step.model_dump()
assert "step_type" in step_dict  # Present!
assert step_dict["step_type"] == "conditional"
```

**Impact:** Serialization round-trips now work correctly. Chain save/load functionality is fixed.

**Files Modified:** `src/mmar_carl/models/steps.py`

---

#### BUG-003 (MEDIUM): Inconsistent Branch Definition Formats

**Problem:** ChainBuilder accepted tuples `("condition", step_number)` but direct construction required `ConditionalBranch` objects, creating API inconsistency.

**Fix:** ChainBuilder now accepts both formats with automatic normalization

**Before:**
```python
# ChainBuilder format - worked
branches=[("contains:positive", 3)]

# Direct construction format - failed
ConditionalStepDescription(
    branches=[("contains:positive", 3)]  # ValidationError!
)
```

**After:**
```python
# Both formats work everywhere!
ChainBuilder().add_conditional_step(
    branches=[("contains:positive", 3)]  # OK
)

ConditionalStepDescription(
    branches=[ConditionalBranch(condition="contains:positive", next_step=3)]  # OK
)

# ChainBuilder also accepts ConditionalBranch objects
ChainBuilder().add_conditional_step(
    branches=[ConditionalBranch(condition="contains:positive", next_step=3)]  # OK
)
```

**Files Modified:** `src/mmar_carl/chain.py`

---

#### BUG-004 (MEDIUM): String Literal Handling in Input Mapping

**Problem:** String literals in input mapping were not properly handled. Quoted strings like `'"value"'` returned `None`.

**Fix:** Added string literal detection to `resolve_context_reference()`

**Before:**
```python
input_mapping={"summary": '"Revenue Analysis"'}
# Result: summary parameter receives None
```

**After:**
```python
input_mapping={"summary": '"Revenue Analysis"'}
# Result: summary parameter receives "Revenue Analysis"
```

**Files Modified:** `src/mmar_carl/step_executors.py`

---

#### BUG-005 (MEDIUM): Type Coercion in Input Mapping

**Problem:** Input mapping didn't perform type coercion from strings to expected parameter types.

**Fix:** Added automatic type coercion based on `ToolParameter.type` field

**Before:**
```python
def calculate_sum(values: list[float]) -> dict:
    return {"sum": sum(values)}

input_mapping={"values": "[100, 200, 300]"}  # String, not list
# Error: unsupported operand type(s) for +: 'int' and 'str'
```

**After:**
```python
input_mapping={"values": "[100, 200, 300]"}
# Result: values parameter receives [100.0, 200.0, 300.0] (list of floats)
```

**Supported Types:** int, float, bool, list, dict

**Files Modified:** `src/mmar_carl/step_executors.py`

---

#### BUG-006 (LOW): Outer Context String Parsing

**Problem:** When `$outer_context` contained structured data as a JSON string, it wasn't parsed before being passed to tools.

**Fix:** Enhanced `$outer_context` handling to parse JSON strings automatically

**Before:**
```python
context = ReasoningContext(
    outer_context="[100, 200, 300]",  # String representation of list
    ...
)
input_mapping={"values": "$outer_context"}
# Error: unsupported operand type(s) for +: 'int' and 'str'
```

**After:**
```python
context = ReasoningContext(
    outer_context="[100, 200, 300]",
    ...
)
input_mapping={"values": "$outer_context"}
# Result: values parameter receives [100, 200, 300] (parsed list)
```

**Files Modified:** `src/mmar_carl/step_executors.py`

---

### 📝 Documentation Improvements

- Updated all examples to remove legacy endpoint/entrypoint references
- Improved import documentation in `__init__.py`
- Enhanced module structure documentation
- Clarified LLM client detection and usage patterns
- Updated README.md to reference separate release notes file

---

### 🗑️ Deprecated

The following features have been completely removed:

- `mmar-llm-mapi` integration code
- `create_llm_client()` function (use `create_openai_client()` instead)
- Legacy endpoint/entrypoint configuration options
- `examples/legacy_mmar_llm_example.py` file

---

### 🔒 Internal Changes

- Refactored LLM client detection logic for better separation of concerns
- Improved error handling in step executors
- Enhanced type safety throughout codebase
- Optimized import structure for better modularity
- Updated `__version__` to "0.2.0"

---

### Migration Guide from v0.1.0

If you're upgrading from v0.1.0, here's what you need to know:

**1. Update your imports (if applicable):**
```python
# If you were importing LLMClientBase directly
from mmar_carl.models.llm_client_base import LLMClientBase
```

**2. Use create_openai_client() for OpenAI-compatible APIs:**
```python
from mmar_carl import create_openai_client

client = create_openai_client(
    api_key="sk-or-v1-...",
    model="anthropic/claude-3.5-sonnet"
)
```

**3. No code changes required for bug fixes!** All bug fixes are backward compatible and will automatically improve your chains.

---

## Version 0.1.0

### 🚨 Deprecation of StepDescription

The unified `StepDescription` class is now **deprecated**. Use typed step classes instead:

```python
# ❌ Deprecated (will show warning)
StepDescription(
    number=1,
    title="Analysis",
    aim="Analyze data"
)

# ✅ Recommended
LLMStepDescription(
    number=1,
    title="Analysis",
    aim="Analyze data"
)
```

### 📊 Structured Logging

New logging system with configurable levels:

```python
import logging
from mmar_carl import set_log_level, get_logger

# Configure logging level
set_log_level(logging.DEBUG)  # or INFO, WARNING, ERROR

# Use logger directly
logger = get_logger()
logger.info("Custom log message")

# Automatic logging during execution:
# 2026-03-10 10:39:09 [INFO] mmar_carl: Starting chain 'My Chain' with 4 steps (max_workers=2)
# 2026-03-10 10:39:18 [INFO] mmar_carl: Chain execution completed successfully in 8.97s (4/4 steps)
# 2026-03-10 10:40:04 [WARNING] mmar_carl: Step 1 failed in 3.16s
```

### 🔍 Error Traceback Preservation

`StepExecutionResult` now includes `error_traceback` field for debugging:

```python
result = chain.execute(context)
if not result.success:
    for step in result.get_failed_steps():
        print(f"Step {step.step_number} failed: {step.error_message}")
        if step.error_traceback:
            print(f"Traceback:\n{step.error_traceback}")
```

### 🛡️ Memory Leak Fix

Context snapshots are now properly cleaned up in parallel execution, preventing event loop issues in long-running applications.

### Chain-Level Timeout

Set a maximum execution time for the entire chain:

```python
chain = ReasoningChain(
    steps=steps,
    timeout=300.0,  # 5 minutes max
)

# Or with ChainBuilder
chain = (ChainBuilder()
    .add_step(...)
    .with_timeout(300.0)
    .build())
```

### Per-Step Retry Configuration

Override retry attempts for specific steps (e.g., more retries for flaky API calls):

```python
LLMStepDescription(
    number=1,
    title="API Call",
    aim="Call external API",
    retry_max=5,  # More retries for this step
)
```

### Resource Cleanup

Properly close LLM clients when done:

```python
context = ReasoningContext(...)
try:
    result = chain.execute(context)
finally:
    await context.close()  # Release HTTP connections
```

### OpenAI-Compatible API Support

Use CARL with OpenRouter, Azure OpenAI, local LLMs (Ollama, vLLM, LM Studio), and any OpenAI-compatible API:

```python
from mmar_carl import create_openai_client, ReasoningContext, Language

# OpenRouter
client = create_openai_client(
    api_key="sk-or-v1-...",
    model="anthropic/claude-3.5-sonnet",
    extra_headers={"HTTP-Referer": "https://your-site.com"}
)

# Local LLM (Ollama)
client = create_openai_client(
    api_key="not-needed",
    model="llama3",
    base_url="http://localhost:11434/v1"
)

context = ReasoningContext(
    outer_context=data,
    api=client,
    language=Language.ENGLISH
)
```

### Per-Step LLM Configuration

Use different models for different reasoning steps:

```python
from mmar_carl import LLMStepDescription, LLMStepConfig

steps = [
    # Fast model for simple tasks
    LLMStepDescription(
        number=1,
        title="Quick Analysis",
        aim="Fast initial analysis",
        # Uses default model from context
    ),
    # Powerful model for complex reasoning
    LLMStepDescription(
        number=2,
        title="Deep Analysis",
        aim="Complex reasoning task",
        llm_config=LLMStepConfig(
            model="anthropic/claude-3.5-sonnet",
            temperature=0.3
        ),
        dependencies=[1]
    ),
]
```

### LLM Execution Modes (Production)

Each LLM step can choose one of two execution strategies via `LLMStepConfig.execution_mode`:

- `ExecutionMode.FAST`: single direct generation (default)
- `ExecutionMode.SELF_CRITIC`: generation with evaluator chain (all evaluators must approve)

```python
from mmar_carl import ExecutionMode, LLMStepConfig, LLMStepDescription
```

#### FAST mode

FAST is strict one-pass generation with no evaluator calls:

```python
LLMStepDescription(
    number=1,
    title="Quick Analysis",
    aim="Generate first-pass answer",
    llm_config=LLMStepConfig(execution_mode=ExecutionMode.FAST),
)
```

#### SELF_CRITIC mode (default LLM evaluator)

SELF_CRITIC runs evaluator(s) after generation. If any evaluator disapproves, the step is regenerated
until all approve or `self_critic_max_revisions` is reached.

Built-in `llm` evaluator behavior:

1. Uses the same step LLM client.
2. Produces strict JSON: `{"verdict":"APPROVE|DISAPPROVE","review":"..."}`.
3. Treats malformed/empty review as `DISAPPROVE`.

```python
LLMStepDescription(
    number=2,
    title="Quality-Controlled Answer",
    aim="Produce higher-quality answer",
    llm_config=LLMStepConfig(
        execution_mode=ExecutionMode.SELF_CRITIC,
        self_critic_evaluators=["llm"],  # built-in evaluator
        self_critic_max_revisions=1,
        self_critic_instruction="Prioritize factual consistency and concrete mitigation advice.",
    ),
)
```

#### Custom self-critic evaluator strategies

You can register custom evaluators by implementing `SelfCriticEvaluatorBase`.

```python
from mmar_carl import SelfCriticEvaluatorBase, SelfCriticDecision

class KeywordGuard(SelfCriticEvaluatorBase):
    async def evaluate(self, step, candidate, base_prompt, context, llm_client, retries):
        if "mitigation" in candidate.lower():
            return SelfCriticDecision("APPROVE", "Keyword present.", {"llm_calls": 0})
        return SelfCriticDecision("DISAPPROVE", "Missing mitigation keyword.", {"llm_calls": 0})

context.register_self_critic_evaluator("keyword_guard", KeywordGuard())
```

Use evaluator chains in a step (all-must-approve policy):

```python
LLMStepDescription(
    number=3,
    title="Final Review",
    aim="Enforce stricter quality rules",
    llm_config=LLMStepConfig(
        execution_mode=ExecutionMode.SELF_CRITIC,
        self_critic_evaluators=["llm", "keyword_guard"],
        self_critic_max_revisions=2,
        self_critic_disapprove_feedback={
            "keyword_guard": "You must explicitly include at least one mitigation item.",
        },
    ),
)
```

#### Execution diagnostics in the pipeline example

`examples/execution_modes_pipeline_example.py` now prints a detailed execution report similar to the
basic example style:

- Chain overview (steps, dependencies, execution plan, per-step mode config)
- Per-step execution results (status, timing, output preview/error)
- Per-step mode diagnostics (`llm_calls`, rounds, evaluator policy, evaluator verdicts by round)
- Final output section

### Chain-Level RE-PLAN Policy

RE-PLAN is a chain-level control policy, not an LLM execution mode.

- `ExecutionMode` defines **how a step runs** (`FAST`, `SELF_CRITIC`).
- `ReplanPolicy` defines **how chain control flow reacts** to intermediate outcomes.

This keeps concerns separate and allows combinations like:

- `FAST` + RE-PLAN
- `SELF_CRITIC` + RE-PLAN
- mixed execution modes + RE-PLAN

#### Minimal RE-PLAN configuration

```python
from mmar_carl import (
    ReplanAction,
    ReplanPolicy,
    RuleBasedReplanCheckerConfig,
    ReasoningChain,
)

policy = ReplanPolicy(
    enabled=True,
    checkers=[
        RuleBasedReplanCheckerConfig(
            name="retry_on_bad_output",
            result_substrings=["needs_retry"],
            action_on_match=ReplanAction.RETRY_CURRENT_STEP,
            feedback_on_match=["Use clearer assumptions and stronger validation."],
        )
    ],
)

chain = ReasoningChain(steps=steps, replan_policy=policy)
```

#### Checker types

- `RuleBasedReplanCheckerConfig`: deterministic/rule-based checks.
- `LLMReplanCheckerConfig`: LLM-based checker with strict structured verdict parsing into `ReplanVerdict`.
- `RegisteredReplanCheckerConfig`: references a custom checker registered in `ReasoningContext`.

#### Aggregation strategies

- `ANY`
- `ALL`
- `K_OF_N`
- `MANDATORY_PLUS_K_OF_REST`

Configure via `ReplanAggregationConfig`.

#### Checkpoints and rollback

Mark any step as a checkpoint with additive step fields:

- `checkpoint=True`
- `checkpoint_name="my_checkpoint"` (optional)

Rollback targets support:

- chain start
- current step
- nearest previous checkpoint
- named checkpoint
- specific step number

#### Triggers, feedback, and safeguards

Configure trigger points with `ReplanTriggerConfig`:

- after step
- after failed step
- checkpoint-only
- selected step numbers/types

Configure loop prevention with `ReplanBudgetConfig`:

- `max_replans_per_chain`
- `max_replans_per_step`
- `max_visits_per_checkpoint`
- repeated same-target protection

RE-PLAN feedback/hints are injected into retried LLM prompts and all replan evaluations/actions are recorded in `ReasoningResult.replan_events` plus summary metadata.

### 📊 Evaluation Metrics

Attach numeric evaluation metrics to individual steps or to the whole chain by subclassing `MetricBase`.

```python
from mmar_carl import MetricBase, LLMStepDescription, ReasoningChain

class WordCountMetric(MetricBase):
    @property
    def name(self) -> str:
        return "word_count"

    async def compute_async(self, text: str) -> float:
        return float(len(text.split()))
```

Attach to a **step** — scored after each step's output:

```python
step = LLMStepDescription(
    number=1,
    title="Analysis",
    aim="Analyse the data",
    metrics=[WordCountMetric()],
)

result = chain.execute(context)
print(result.step_results[0].metrics)   # {'word_count': 47.0}
```

Attach to the **chain** — scored on the final output:

```python
chain = ReasoningChain(steps=steps, metrics=[WordCountMetric()])
result = chain.execute(context)
print(result.metrics)                   # {'word_count': 82.0}
```

Scores are stored in `StepExecutionResult.metrics` and `ReasoningResult.metrics`, included in `to_dict()`, and sent to LangFuse (step scores appear in span output; chain scores are posted as LangFuse `score` objects).

**Key properties:**

- Any number of metrics on the same step or chain
- Metrics run only on **successful** outputs; failed steps are skipped
- A metric that raises an exception is silently skipped — it never aborts execution
- Implement `compute_async(text) -> float`; a sync `compute()` wrapper is provided for convenience

**Built-in examples** (`examples/metrics_example.py` — no API key needed):

| Metric | What it measures |
|---|---|
| `WordCountMetric` | Number of words in the output |
| `SentenceLengthMetric` | Average words per sentence |
| `KeywordCoverageMetric(keywords)` | Fraction of required keywords present (0–1) |
| `MockLLMJudgeMetric` | Simulated LLM-as-a-judge score (0–10) |

Run the example:

```bash
python examples/metrics_example.py
# or
make example-metrics
```

#### Metrics in Reflection

When calling `chain.reflect()`, metric scores are automatically fed into the reflection prompt so the LLM can reference concrete quality signals:

```python
from mmar_carl import ReflectionOptions

result = chain.execute(context)

reflection = chain.reflect(
    task_description="Analyse quarterly revenue",
    options=ReflectionOptions(
        include_metric_scores=True,      # True by default
        extra_feedback={                 # optional user context
            "audience": "C-level executives",
            "priority": "conciseness",
        },
    ),
)
```

`extra_feedback` accepts a `dict` (labelled entries) or a plain `str`. Set `include_metric_scores=False` to exclude scores from the prompt.

```bash
python examples/reflection_metrics_example.py
# or
make example-reflection-metrics
```

### Typed Step Description Classes

New inheritance-based step classes for better type safety:

- `LLMStepDescription` - LLM reasoning steps
- `ToolStepDescription` - External tool/function execution
- `MCPStepDescription` - MCP protocol calls
- `MemoryStepDescription` - Memory operations
- `TransformStepDescription` - Data transformations
- `ConditionalStepDescription` - Conditional branching

### Multi-Step Type Support

Execute different types of operations in your reasoning chains:

- **LLM**: Standard LLM reasoning (default)
- **TOOL**: Execute registered Python functions
- **MCP**: Call MCP protocol servers
- **MEMORY**: Read/write/append/delete/list operations
- **TRANSFORM**: Data transformations without LLM
- **CONDITIONAL**: Branch execution based on conditions

### Memory and Tool Registry

- Built-in memory storage with namespace isolation
- Tool registry for registering external functions
- Input mapping syntax: `$history[-1]`, `$memory.namespace.key`, `$metadata.key`

### JSON Serialization

- `chain.save("file.json")` / `ReasoningChain.load("file.json")`
- `chain.to_dict()` / `ReasoningChain.from_dict(data)`
- `chain.to_json()` / `ReasoningChain.from_json(json_str)`
