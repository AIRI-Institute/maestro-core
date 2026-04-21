# MMAR CARL - Collaborative Agent Reasoning Library

A Python library for building universal chain-of-thought reasoning systems with RAG-like context extraction and DAG-based parallel execution.

## Overview

CARL provides a structured framework for creating expert chain-of-thought reasoning systems that can execute steps in parallel where dependencies allow. It features **RAG-like context querying** that automatically extracts relevant information from the input data for each reasoning step. Designed to help developers implement sophisticated expert reasoning chains in their AI agents with support for any domain and multi-language capabilities (Russian/English).

## Key Features

- **🔍 Advanced Context Extraction**: Configurable search strategies (substring and FAISS vector search) for intelligent context retrieval
- **🎯 Per-Query Search Configuration**: Fine-grained control with individual search strategy overrides for each query
- **⚡ DAG-based Execution**: Automatically parallelizes reasoning steps based on dependencies
- **🤖 Automatic LLM Client Detection**: Smart detection of LLMHub with automatic client creation
- **🌐 OpenAI-Compatible APIs**: Support for OpenRouter, Azure OpenAI, local LLMs (Ollama, vLLM, LM Studio)
- **🎛️ Per-Step LLM Configuration**: Use different models for different reasoning steps
- **🎛️ System Prompt Support**: Include domain-specific instructions and persona in every reasoning step
- **🔗 Direct mmar-llm Integration**: Seamless integration with LLMHub
- **🌍 Multi-language Support**: Built-in support for Russian and English languages with easy extensibility
- **🏗️ Universal Architecture**: Works with any domain - financial, medical, legal, technical, or custom expert knowledge
- **⚙️ Production Ready**: Async/sync compatibility, error handling, and retry logic
- **🚀 Parallel Processing**: Optimized execution with configurable worker pools
- **🎯 Expert Reasoning**: Designed for implementing sophisticated chain-of-thought reasoning in AI agents
- **🔧 Flexible Search**: Choose between fast substring search or advanced vector search with semantic similarity
- **🔄 Mixed Search Strategies**: Combine different search methods within the same reasoning step
- **🛠️ Multi-Step Types**: Support for LLM, Tool, MCP, Memory, Transform, and Conditional steps
- **💾 Memory Operations**: Built-in memory storage with namespaces for state management
- **🔌 Tool Integration**: Register and execute external functions within reasoning chains
- **📦 JSON Serialization**: Save and load chains to/from JSON files

## Installation

### Basic Installation

For basic usage with substring search (minimal dependencies):

```bash
pip install mmar-carl
```

### With Vector Search

For vector-based context extraction with semantic similarity:

```bash
pip install 'mmar-carl[vector-search]'
```

### With All Features

Install all optional dependencies:

```bash
pip install 'mmar-carl[all]'
```

### Feature-Specific Installs

```bash
# MCP support
pip install 'mmar-carl[mcp]'

# OpenAI-compatible APIs
pip install 'mmar-carl[openai]'

# Langfuse tracing
pip install 'mmar-carl[langfuse]'
```

### Choosing Search Strategy

- **Substring search** (default): Fast, no additional dependencies, good for exact keyword matching
- **Vector search**: Semantic similarity, requires `faiss-cpu` + `fastembed` + `numpy`, better for contextual matching

```python
# Basic usage (no extra dependencies)
from mmar_carl.models.search import ContextSearchConfig

config = ContextSearchConfig(strategy="substring")

# Advanced usage (requires vector-search)
config = ContextSearchConfig(
    strategy="vector",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
```


## Release Notes

For detailed information about each version, see [RELEASE.md](RELEASE.md)

### Quick Links

- **[v0.2.0](RELEASE.md#version-02-0---2026-04-15)** - Latest Release
  - 🔧 Breaking Changes: LLM client refactoring, removed mmar-llm-mapi
  - ✨ New Features: 74 new tests, examples runner
  - 🐛 Bug Fixes: 6 critical bugs fixed (conditional steps, serialization, input mapping)
  - 📝 Documentation improvements throughout

- **[v0.1.0](RELEASE.md#version-01-0)** - Previous Release
  - 🚨 Typed step description classes (StepDescription deprecated)
  - 📊 Structured logging with configurable levels
  - 🔍 Error traceback preservation
  - 🛡️ Memory leak fix
  - ⏱️ Chain-level timeout
  - 🔄 Per-step retry configuration
  - 🌐 OpenAI-compatible API support
  - 🎛️ Per-step LLM configuration
  - ⚡ LLM execution modes (FAST/SELF_CRITIC)
  - 🔀 Chain-level RE-PLAN policy
  - 📊 Evaluation metrics

## Quick Start

```python
import asyncio
from mmar_carl import (
    ReasoningChain, StepDescription, ReasoningContext,
    Language
)
from mmar_llm import LLMHub, LLMConfig

# Define a reasoning chain with RAG-like context queries
EXPERT_ANALYSIS = [
    StepDescription(
        number=1,
        title="Initial Data Assessment",
        aim="Assess the quality and completeness of input data",
        reasoning_questions="What data patterns and anomalies are present?",
        step_context_queries=["data quality indicators", "missing values", "data consistency"],
        stage_action="Evaluate data reliability and identify potential issues",
        example_reasoning="High-quality data enables more reliable analysis and predictions"
    ),
    StepDescription(
        number=2,
        title="Pattern Recognition",
        aim="Identify significant patterns and trends in the data",
        reasoning_questions="What trends and correlations emerge from the analysis?",
        dependencies=[1],  # Depends on data quality assessment
        step_context_queries=["growth trends", "performance indicators", "correlation patterns"],
        stage_action="Analyze temporal patterns and statistical relationships",
        example_reasoning="Pattern recognition helps identify underlying business drivers and opportunities"
    )
]

# Create and execute the reasoning chain
chain = ReasoningChain(
    steps=EXPERT_ANALYSIS,
    max_workers=2,
    enable_progress=True
)

# Context with data (CSV, JSON, text, or any domain-specific data)
data_context = """
Period,Revenue,Profit,Employees
2023-Q1,1000000,200000,50
2023-Q2,1200000,300000,55
2023-Q3,1100000,250000,52
2023-Q4,1400000,400000,60
"""

context = ReasoningContext(
    outer_context=data_context,
    language=Language.ENGLISH,
    retry_max=3,
    system_prompt="You are a senior data analyst with expertise in financial data interpretation."
)

result = chain.execute(context)
print(result.get_final_output())
```

## Migration Guide: From StepDescription to Typed Classes

If you're using the deprecated `StepDescription` class, migrate to typed classes for better type safety and clearer intent:

### LLM Steps

```python
# ❌ Old (deprecated)
StepDescription(
    number=1,
    title="Analysis",
    aim="Analyze data",
    reasoning_questions="What patterns exist?",
    step_type=StepType.LLM
)

# ✅ New
LLMStepDescription(
    number=1,
    title="Analysis",
    aim="Analyze data",
    reasoning_questions="What patterns exist?"
    # No need to specify step_type - it's implicit
)
```

### Tool Steps

```python
# ❌ Old
StepDescription(
    number=1,
    title="Calculate",
    step_type=StepType.TOOL,
    step_config=ToolStepConfig(tool_name="calc", input_mapping={})
)

# ✅ New
ToolStepDescription(
    number=1,
    title="Calculate",
    config=ToolStepConfig(tool_name="calc", input_mapping={})
)
```

### Memory Steps

```python
# ❌ Old
StepDescription(
    number=1,
    title="Store",
    step_type=StepType.MEMORY,
    step_config=MemoryStepConfig(operation=MemoryOperation.WRITE, ...)
)

# ✅ New
MemoryStepDescription(
    number=1,
    title="Store",
    config=MemoryStepConfig(operation=MemoryOperation.WRITE, ...)
)
```

### Conversion Helper

Use `to_typed_step()` to convert legacy steps:

```python
legacy_step = StepDescription(number=1, title="Old", aim="Old step")
typed_step = legacy_step.to_typed_step()  # Returns LLMStepDescription
```

### Benefits of Typed Classes

- ✅ **Type Safety**: IDE autocomplete and type checking
- ✅ **Clear Intent**: Class name indicates step type
- ✅ **Validation**: Better error messages for missing fields
- ✅ **Future-Proof**: New features will be added to typed classes first

## Using Typed Step Classes (New API)

The new typed API provides better type safety and clearer intent:

```python
from mmar_carl import (
    ReasoningChain, LLMStepDescription, ToolStepDescription,
    MemoryStepDescription, ReasoningContext, Language,
    ToolStepConfig, MemoryStepConfig, MemoryOperation
)

# Define a chain with mixed step types
steps = [
    # LLM reasoning step
    LLMStepDescription(
        number=1,
        title="Analyze Data",
        aim="Analyze the input data for patterns",
        reasoning_questions="What trends and patterns exist?",
        stage_action="Extract key insights",
        example_reasoning="Pattern analysis reveals growth trends"
    ),
    # Tool execution step
    ToolStepDescription(
        number=2,
        title="Calculate Metrics",
        config=ToolStepConfig(
            tool_name="calculate_growth",
            input_mapping={"data": "$history[-1]"}
        ),
        dependencies=[1]
    ),
    # Memory storage step
    MemoryStepDescription(
        number=3,
        title="Store Results",
        config=MemoryStepConfig(
            operation=MemoryOperation.WRITE,
            memory_key="analysis_result",
            value_source="$history[-1]",
            namespace="results"
        ),
        dependencies=[2]
    ),
]

chain = ReasoningChain(steps=steps, max_workers=2)

# Register tools before execution
def calculate_growth(data: str) -> dict:
    return {"growth_rate": 0.15, "trend": "positive"}

context = ReasoningContext(
    outer_context=data,
    api=llm_hub,
    language=Language.ENGLISH
)
context.register_tool("calculate_growth", calculate_growth)

result = chain.execute(context)
```

## Using ChainBuilder with Multiple Step Types

```python
from mmar_carl import ChainBuilder

chain = (ChainBuilder()
    # LLM step
    .add_step(
        number=1,
        title="Analysis",
        aim="Analyze data",
        reasoning_questions="What patterns exist?",
        stage_action="Extract insights",
        example_reasoning="Example"
    )
    # Tool step
    .add_tool_step(
        number=2,
        title="Calculate",
        tool_name="my_calculator",
        input_mapping={"value": "$history[-1]"},
        dependencies=[1]
    )
    # Memory write step
    .add_memory_step(
        number=3,
        title="Store",
        operation="write",
        memory_key="result",
        value_source="$history[-1]",
        dependencies=[2]
    )
    # Memory read step
    .add_memory_step(
        number=4,
        title="Retrieve",
        operation="read",
        memory_key="result",
        dependencies=[3]
    )
    .with_max_workers(2)
    .build())
```

## JSON Serialization

Save and load chains for reuse:

```python
# Save chain to file
chain.save("my_chain.json")

# Load chain from file
loaded_chain = ReasoningChain.load("my_chain.json")

# Convert to/from dict
chain_dict = chain.to_dict()
restored_chain = ReasoningChain.from_dict(chain_dict)

# Convert to/from JSON string
json_str = chain.to_json()
restored_chain = ReasoningChain.from_json(json_str)
```

## Automatic LLM Client Detection

CARL features **intelligent LLM client detection** that automatically creates the appropriate client based on your API object. Simply pass an `LLMHub` instance, and CARL will handle the rest:

```python
from mmar_carl import ReasoningContext, Language
from mmar_llm import LLMHub, LLMConfig
from mmar_mapi.api import LLMHubAPI
from mmar_ptag import ptag_client  # For PTAG-generated clients

# Option 1: With LLMHub from mmar-llm
llm_hub = LLMHub(config)
context = ReasoningContext(
    outer_context=data,
    api=llm_hub,
    language=Language.ENGLISH
)

# Option 2: With LLMHubAPI from mmar-mapi
llm_api = LLMHubAPI()
context = ReasoningContext(
    outer_context=data,
    api=llm_api,
    language=Language.ENGLISH
)

# Option 3: With PTAG-generated client
ptag_client_instance = ptag_client(LLMHub, "localhost:50051")
context = ReasoningContext(
    outer_context=data,
    api=ptag_client_instance,
    language=Language.ENGLISH
)
```

### Supported API Types

- **OpenAICompatibleClient**: OpenRouter, Azure OpenAI, local LLMs (Ollama, vLLM, LM Studio)
- **LLMHub**: Direct integration with mmar-llm library
- **LLMHubAPI**: Integration with mmar-mapi library
- **PTAG Clients**: Dynamically created clients via `ptag_client()`
- **Mock Objects**: Test implementations that simulate the interface
- **Duck Typing**: Any object implementing `__getitem__` or `get_response` methods

The detection works by analyzing the interface capabilities and type names to determine the most appropriate LLM client to create.

## System Prompt Support

CARL supports **system prompts** that allow you to provide consistent instructions and persona across all reasoning steps. This is particularly useful for domain-specific expertise and maintaining consistent behavior throughout complex reasoning chains.

```python
# Define domain expertise through system prompt
financial_system_prompt = """
You are a senior financial analyst with 15 years of experience in corporate finance.

Your analysis should:
- Be data-driven and evidence-based
- Include specific percentages and trends
- Provide actionable insights and recommendations
- Consider industry benchmarks and best practices
- Maintain professional objectivity
"""

context = ReasoningContext(
    outer_context=financial_data,
    language=Language.ENGLISH,
    system_prompt=financial_system_prompt.strip()
)
```

### System Prompt Benefits

- **🎛️ Consistent Persona**: Apply expert personality to all reasoning steps
- **🏥 Domain Expertise**: Inject specialized knowledge (medical, legal, financial, etc.)
- **🌍 Multi-language Support**: System prompts work in both English and Russian
- **⚡ Parallel Execution**: System prompt is preserved across all parallel steps
- **🔧 Flexible Configuration**: Optional field that defaults to empty string for backward compatibility

### System Prompt Format

System prompts are automatically prefixed to each reasoning step prompt:

**English:**

```
System Instructions:
You are a senior financial analyst with 15 years of experience...

Data for analysis:
[regular chain prompt content]
```

**Russian:**

```
Системные инструкции:
Вы старший финансовый аналитик с 15-летним опытом...

Данные для анализа:
[regular chain prompt content]
```

## Installation

```bash
# For production use
pip install mmar-carl

# With OpenAI-compatible API support (OpenRouter, Azure, local LLMs)
pip install 'mmar-carl[openai]'

# For development with mmar-llm integration
pip install mmar-carl mmar-llm~=2.0.11

# Development version with all dependencies
pip install mmar-carl[dev]

# With optional vector search capabilities (FAISS)
pip install mmar-carl[search]

# Or install search dependencies manually
pip install mmar-carl faiss-cpu>=1.7.0 numpy>=1.21.0 sentence-transformers>=2.2.0
```

## Requirements

- Python 3.12+
- mmar-llm~=2.0.11 (for LLM integration)
- Pydantic for data models
- asyncio for parallel execution

**Optional Dependencies for Advanced Search:**

- faiss-cpu>=1.7.0 (for vector search)
- numpy>=1.21.0 (for vector operations)
- sentence-transformers>=2.2.0 (for embeddings)

## Documentation

- **Reasoning Methodology**: [docs/REASONING.md](docs/REASONING.md) - Basic reasoning chains methodology (in Russian)
- **Advanced Reasoning**: [docs/REASONING+.md](docs/REASONING+.md) - Advanced reasoning chains with detailed examples (in Russian)

## Architecture

CARL is built around several key components:

### Core Components

- **ReasoningChain**: Orchestrates the execution of reasoning steps with DAG optimization and JSON serialization
- **DAGExecutor**: Handles parallel execution based on dependencies with configurable workers
- **ReasoningContext**: Manages execution state, history, memory storage, tool registry, and LLM client

### Parallel Execution Notes

When running steps in parallel:

- **Memory isolation**: Each parallel step gets a deep copy of memory. Writes are NOT visible to parallel siblings.
- **Tool safety**: Tools are shared via shallow copy. Tools MUST be stateless for thread safety.
- **Visibility**: Memory writes from parallel steps become visible only to subsequent batches.

### Conditional Steps (Experimental)

CONDITIONAL steps return `next_step` in `result_data` but the DAG executor currently proceeds in topological order. For true branching behavior, consider:

- Using separate chains for different paths
- Designing dependencies to skip unwanted branches

### Step Description Classes

- **StepDescriptionBase**: Abstract base class for all step types
- **LLMStepDescription**: LLM reasoning steps with aim, questions, and context queries
- **ToolStepDescription**: External tool/function execution
- **MCPStepDescription**: MCP protocol server calls
- **MemoryStepDescription**: Memory operations (read/write/append/delete/list)
- **TransformStepDescription**: Data transformations without LLM
- **ConditionalStepDescription**: Conditional branching logic
- **StepDescription**: Legacy unified class (backward compatible)

### MCP Step Executor (Experimental)

MCP steps allow calling Model Context Protocol servers. Note: This is experimental and requires the `mcp` package:

```python
MCPStepDescription(
    number=1,
    title="MCP Tool Call",
    config=MCPStepConfig(
        server=MCPServerConfig(
            server_name="my_server",
            command="python",
            args=["-m", "my_mcp_server"],
        ),
        tool_name="my_tool",
    ),
)
```

For production use, consider registering MCP tools as regular Python functions via `context.register_tool()`.

### Step Executors

- **LLMStepExecutor**: Executes LLM reasoning with prompt generation
- **ToolStepExecutor**: Executes registered Python functions
- **MCPStepExecutor**: Handles MCP protocol communication
- **MemoryStepExecutor**: Manages memory operations
- **TransformStepExecutor**: Performs data transformations
- **ConditionalStepExecutor**: Evaluates conditions and branches

### Supporting Components

- **LLMClientFactory**: Auto-detects API types (LLMHub, LLMHubAPI)
- **Language**: Built-in support for Russian and English
- **PromptTemplate**: Multi-language prompt templates
- **SearchStrategy**: Substring and vector search for context extraction

## Key Concepts

### DAG-Based Parallel Execution

CARL automatically analyzes step dependencies and creates execution batches for maximum parallelization:

```python
# Steps 1 and 2 execute in parallel
StepDescription(number=1, title="Revenue Analysis", dependencies=[])
StepDescription(number=2, title="Cost Analysis", dependencies=[])
# Step 3 waits for both to complete
StepDescription(number=3, title="Profitability Analysis", dependencies=[1, 2])
```

### RAG-like Context Extraction

Automatically extracts relevant context from input data for each reasoning step:

```python
# Define context queries to extract relevant information
step = StepDescription(
    number=1,
    title="Financial Analysis",
    aim="Analyze financial performance",
    reasoning_questions="What are the key financial trends?",
    step_context_queries=["revenue growth", "profit margins", "cost efficiency"],
    stage_action="Calculate financial ratios and trends",
    example_reasoning="Financial analysis reveals business health and performance drivers"
)

# CARL automatically extracts relevant context from outer_context
# For each query, it searches the input data and includes findings in the LLM prompt
```

### Multi-language Support

Built-in support for Russian and English with appropriate prompt templates:

```python
# Russian language reasoning
context = ReasoningContext(
    outer_context=data,
    language=Language.RUSSIAN,
    system_prompt="Вы экспертный финансовый аналитик с профессиональным опытом."
)

# English language reasoning
context = ReasoningContext(
    outer_context=data,
    language=Language.ENGLISH,
    system_prompt="You are an expert financial analyst with professional experience."
)
```

### Advanced Search Configuration

CARL supports multiple search strategies for context extraction:

#### Substring Search (Default)

Simple, fast text-based search that works without additional dependencies:

```python
from mmar_carl import ContextSearchConfig, ReasoningChain

# Configure case-sensitive substring search
search_config = ContextSearchConfig(
    strategy="substring",
    substring_config={
        "case_sensitive": True,
        "min_word_length": 3,
        "max_matches_per_query": 5
    }
)

chain = ReasoningChain(
    steps=steps,
    search_config=search_config
)
```

#### Vector Search with FAISS

Advanced semantic search using embeddings and vector similarity:

```python
# Configure vector search with FAISS
search_config = ContextSearchConfig(
    strategy="vector",
    embedding_model="all-MiniLM-L6-v2",  # Optional: custom model
    vector_config={
        "index_type": "flat",  # or "ivf" for large datasets
        "similarity_threshold": 0.7,
        "max_results": 5
    }
)

chain = ReasoningChain(
    steps=steps,
    search_config=search_config
)
```

#### Per-Query Search Configuration

For fine-grained control, you can specify different search strategies for individual queries:

```python
from mmar_carl import ContextQuery, StepDescription

# Mix of string queries and ContextQuery objects in the same step
step = StepDescription(
    number=1,
    title="Advanced Analysis",
    aim="Analyze with mixed search strategies",
    reasoning_questions="What insights can we extract?",
    stage_action="Extract comprehensive insights",
    example_reasoning="Mixed search provides comprehensive analysis",
    step_context_queries=[
        "EBITDA",  # Simple string (uses chain default)
        ContextQuery(
            query="revenue trends",
            search_strategy="vector",
            search_config={
                "similarity_threshold": 0.8,
                "max_results": 3
            }
        ),
        ContextQuery(
            query="NET_INCOME",
            search_strategy="substring",
            search_config={
                "case_sensitive": True,
                "min_word_length": 4
            }
        )
    ]
)
```

#### Using the ChainBuilder with Search Configuration

```python
from mmar_carl import ChainBuilder, ContextSearchConfig

search_config = ContextSearchConfig(
    strategy="vector",
    vector_config={"similarity_threshold": 0.8}
)

chain = (ChainBuilder()
    .add_step(
        number=1,
        title="Analysis Step",
        aim="Analyze data patterns",
        reasoning_questions="What patterns emerge?",
        stage_action="Extract insights",
        example_reasoning="Pattern analysis reveals trends",
        step_context_queries=["performance metrics", "trends", "anomalies"]
    )
    .with_search_config(search_config)
    .with_max_workers(2)
    .build())
```

### Automatic LLM Client Integration

Simple and straightforward usage with automatic client detection:

```python
from mmar_llm import LLMHub
from mmar_mapi.api import LLMHubAPI

# Automatic usage pattern - works with both API types
context = ReasoningContext(
    outer_context=data,
    api=llm_hub
)

# Also works with LLMHubAPI
context = ReasoningContext(
    outer_context=data,
    api=llm_api
)
```

## Logging and Debugging

### Structured Logging

CARL includes a built-in logging system for monitoring chain execution:

```python
import logging
from mmar_carl import set_log_level, get_logger

# Enable debug logging for development
set_log_level(logging.DEBUG)

# Or use INFO for production (default)
set_log_level(logging.INFO)

# Custom logging
logger = get_logger()
logger.info("Starting custom analysis")
logger.debug("Processing %d steps", len(steps))
```

### Log Output Examples

```
2026-03-10 10:39:09 [INFO] mmar_carl: Starting chain 'Financial Analysis' with 3 steps (max_workers=2)
2026-03-10 10:39:12 [DEBUG] mmar_carl: Starting step 1: Data Assessment (type=StepType.LLM)
2026-03-10 10:39:15 [DEBUG] mmar_carl: Step 1 completed in 3.21s
2026-03-10 10:39:15 [DEBUG] mmar_carl: Executing batch 2 with 2 steps in parallel
2026-03-10 10:39:18 [WARNING] mmar_carl: Step 2 failed in 2.89s
2026-03-10 10:39:18 [INFO] mmar_carl: Chain execution failed in 9.15s (2/3 steps)
```

### Error Handling with Traceback

Failed steps include full traceback for debugging:

```python
result = chain.execute(context)

if not result.success:
    print(f"Chain failed: {len(result.get_failed_steps())} steps failed")

    for step in result.get_failed_steps():
        print(f"\nStep {step.step_number}: {step.title}")
        print(f"Error: {step.error_message}")

        if step.error_traceback:
            print(f"\nFull traceback:\n{step.error_traceback}")
```

### Log Level Reference

| Level   | When to Use                                |
| ------- | ------------------------------------------ |
| DEBUG   | Development, detailed execution flow       |
| INFO    | Production, chain start/complete (default) |
| WARNING | Production, failed steps                   |
| ERROR   | Critical errors                            |

## Example Usage

See the [examples/](examples/) directory for comprehensive examples:

- **[basic_chain_example.py](examples/basic_chain_example.py)**: Core concepts, chain creation, dependencies, serialization
- **[openrouter_example.py](examples/openrouter_example.py)**: OpenRouter/OpenAI-compatible API integration
- **[tool_steps_example.py](examples/tool_steps_example.py)**: Tool steps, memory operations, mixed chains
- **[structured_output_example.py](examples/structured_output_example.py)**: Schema-constrained JSON output via Pydantic models
- **[llm_council_example.py](examples/llm_council_example.py)**: LLM Council — multiple models voting on a decision in parallel
- **[reflection_example.py](examples/reflection_example.py)**: Reflection feature for analyzing chain execution results
- **[execution_modes_pipeline_example.py](examples/execution_modes_pipeline_example.py)**: FAST + SELF_CRITIC with real API + detailed diagnostics output
- **[execution_modes_mock_example.py](examples/execution_modes_mock_example.py)**: FAST + SELF_CRITIC with local mock client
- **[metrics_example.py](examples/metrics_example.py)**: Evaluation metrics — word count, keyword coverage, LLM-as-a-judge; no API key needed
- **[reflection_metrics_example.py](examples/reflection_metrics_example.py)**: Metric scores + `extra_feedback` fed into reflection prompt; no API key needed

### Running Examples

```bash
# Basic chain examples (requires API key for execution)
export OPENAI_API_KEY="sk-or-v1-..."
python examples/basic_chain_example.py

# OpenRouter examples (requires API key)
python examples/openrouter_example.py

# Tool and memory examples (tool-only examples run without API key)
python examples/tool_steps_example.py

# Structured output examples (requires API key)
python examples/structured_output_example.py

# LLM Council — multi-model voting (requires API key)
python examples/llm_council_example.py

# Reflection — analyze chain execution results (requires API key)
python examples/reflection_example.py

# Execution modes pipeline with real API (requires API key, prints detailed report)
python examples/execution_modes_pipeline_example.py

# Execution modes pipeline with a local mock client (no API key)
python examples/execution_modes_mock_example.py

# Evaluation metrics — no API key needed
python examples/metrics_example.py

# Metric-enhanced reflection — no API key needed
python examples/reflection_metrics_example.py
```

## 🚀 Perfect for AI Agent Development

CARL is designed specifically for developers building sophisticated AI agents:

- **🎯 Expert Reasoning Chains**: Implement domain-expert thinking processes
- **🏥 Medical Analysis**: Clinical decision support systems
- **⚖️ Legal Reasoning**: Case analysis and legal document processing
- **💰 Financial Intelligence**: Investment analysis and risk assessment
- **🔬 Scientific Research**: Data analysis and hypothesis testing
- **🏭 Business Intelligence**: Market analysis and strategic planning
- **And any domain requiring structured expert reasoning**

## Universal and Extensible

- **🔧 Customizable**: Works with any data format (CSV, JSON, text, logs, etc.)
- **🌐 Language Agnostic**: Easy to add support for any language
- **📚 Domain Flexible**: Adaptable to any expert domain or industry
- **🔗 Integration Ready**: Works with any LLM provider via mmar-llm
- **⚡ Production Ready**: Built for real-world applications
