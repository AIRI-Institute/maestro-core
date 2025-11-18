# MMAR CARL - Collaborative Agent Reasoning Library

A Python library for building universal chain-of-thought reasoning systems with RAG-like context extraction and DAG-based parallel execution.

## Overview

CARL provides a structured framework for creating expert chain-of-thought reasoning systems that can execute steps in parallel where dependencies allow. It features **RAG-like context querying** that automatically extracts relevant information from the input data for each reasoning step. Designed to help developers implement sophisticated expert reasoning chains in their AI agents with support for any domain and multi-language capabilities (Russian/English).

## Key Features

- **🔍 Advanced Context Extraction**: Configurable search strategies (substring and FAISS vector search) for intelligent context retrieval
- **🎯 Per-Query Search Configuration**: Fine-grained control with individual search strategy overrides for each query
- **⚡ DAG-based Execution**: Automatically parallelizes reasoning steps based on dependencies
- **🤖 Automatic LLM Client Detection**: Smart detection of EntrypointsAccessor or LLMAccessorAPI with automatic client creation
- **🎛️ System Prompt Support**: Include domain-specific instructions and persona in every reasoning step
- **🔗 Direct mmar-llm Integration**: Seamless integration with EntrypointsAccessor and LLMAccessorAPI
- **🌍 Multi-language Support**: Built-in support for Russian and English languages with easy extensibility
- **🏗️ Universal Architecture**: Works with any domain - financial, medical, legal, technical, or custom expert knowledge
- **⚙️ Production Ready**: Async/sync compatibility, error handling, and retry logic
- **🚀 Parallel Processing**: Optimized execution with configurable worker pools
- **🎯 Expert Reasoning**: Designed for implementing sophisticated chain-of-thought reasoning in AI agents
- **🔧 Flexible Search**: Choose between fast substring search or advanced vector search with semantic similarity
- **🔄 Mixed Search Strategies**: Combine different search methods within the same reasoning step

## Quick Start

```python
import asyncio
from mmar_carl import (
    ReasoningChain, StepDescription, ReasoningContext,
    Language
)
from mmar_llm import EntrypointsAccessor, EntrypointsConfig

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

# Create entrypoints accessor from configuration file
def create_entrypoints(entrypoints_path: str):
    """Create EntrypointsAccessor from configuration file."""
    import json
    with open(entrypoints_path, encoding="utf-8") as f:
        config_data = json.load(f)

    entrypoints_config = EntrypointsConfig.model_validate(config_data)
    return EntrypointsAccessor(entrypoints_config)

# Create and execute the reasoning chain
entrypoints = create_entrypoints("entrypoints.json")
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
    api=entrypoints,  # Automatic LLM client detection
    entrypoint_key="my_entrypoint",
    language=Language.ENGLISH,
    retry_max=3,
    system_prompt="You are a senior data analyst with expertise in financial data interpretation."
)

result = chain.execute(context)
print(result.get_final_output())
```

## Automatic LLM Client Detection

CARL features **intelligent LLM client detection** that automatically creates the appropriate client based on your API object. Simply pass either an `EntrypointsAccessor` or `LLMAccessorAPI` instance, and CARL will handle the rest:

```python
from mmar_carl import ReasoningContext, Language
from mmar_llm import EntrypointsAccessor
from mmar_mapi.api import LLMAccessorAPI
from mmar_ptag import ptag_client  # For PTAG-generated clients

# Option 1: With EntrypointsAccessor
entrypoints = EntrypointsAccessor(config)
context = ReasoningContext(
    outer_context=data,
    api=entrypoints,  # Automatically creates EntrypointsAccessorLLMClient
    entrypoint_key="my_entrypoint",
    language=Language.ENGLISH
)

# Option 2: With LLMAccessorAPI
llm_api = LLMAccessorAPI()
context = ReasoningContext(
    outer_context=data,
    api=llm_api,  # Automatically creates LLMAccessorClient
    entrypoint_key="my_entrypoint",
    language=Language.ENGLISH
)

# Option 3: With PTAG-generated client
ptag_client_instance = ptag_client(LLMAccessorAPI, "localhost:50051")
context = ReasoningContext(
    outer_context=data,
    api=ptag_client_instance,  # Automatically detects and creates LLMAccessorClient
    entrypoint_key="my_entrypoint",
    language=Language.ENGLISH
)
```

### Supported API Types

- **EntrypointsAccessor**: Direct integration with mmar-llm library
- **LLMAccessorAPI**: Integration with mmar-mapi library
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
    api=entrypoints,
    entrypoint_key="my_entrypoint",
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

# For development with mmar-llm integration
pip install mmar-carl mmar-llm>=1.0.3

# Development version with all dependencies
pip install mmar-carl[dev]

# With optional vector search capabilities (FAISS)
pip install mmar-carl[search]

# Or install search dependencies manually
pip install mmar-carl faiss-cpu>=1.7.0 numpy>=1.21.0 sentence-transformers>=2.2.0
```

## Requirements

- Python 3.12+
- mmar-llm>=1.0.3 (for LLM integration)
- Pydantic for data models
- asyncio for parallel execution

**Optional Dependencies for Advanced Search:**
- faiss-cpu>=1.7.0 (for vector search)
- numpy>=1.21.0 (for vector operations)
- sentence-transformers>=2.2.0 (for embeddings)

## Documentation

- **Quick Start**: [docs/quickstart.md](docs/quickstart.md) - Get up and running quickly
- **Examples**: [docs/examples.md](docs/examples.md) - Real-world usage examples
- **Advanced Usage**: [docs/advanced.md](docs/advanced.md) - Advanced features and optimization
- **Methodology**: [docs/methodology.md](docs/methodology.md) - Development methodology (in Russian)

## Architecture

CARL is built around several key components:

- **StepDescription**: Defines individual reasoning steps with metadata, dependencies, and RAG-like context queries
- **ReasoningChain**: Orchestrates the execution of reasoning steps with DAG optimization
- **DAGExecutor**: Handles parallel execution based on dependencies with configurable workers
- **ReasoningContext**: Manages execution state, history, multi-language support, and input data with automatic LLM client detection
- **LLMClientFactory**: Automatically detects API types and creates appropriate LLM clients (EntrypointsAccessorLLMClient or LLMAccessorClient)
- **Language**: Built-in support for Russian and English languages (easily extensible)
- **PromptTemplate**: Multi-language prompt templates with RAG-like context integration

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
    api=entrypoints,  # Automatic LLM client detection
    entrypoint_key="my_entrypoint",
    language=Language.RUSSIAN,
    system_prompt="Вы экспертный финансовый аналитик с профессиональным опытом."
)

# English language reasoning
context = ReasoningContext(
    outer_context=data,
    api=entrypoints,  # Automatic LLM client detection
    entrypoint_key="my_entrypoint",
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
from mmar_llm import EntrypointsAccessor
from mmar_mapi.api import LLMAccessorAPI

# Automatic usage pattern - works with both API types
context = ReasoningContext(
    outer_context=data,
    api=entrypoints,  # EntrypointsAccessor - creates EntrypointsAccessorLLMClient
    entrypoint_key="my_entrypoint"
)

# Also works with LLMAccessorAPI
context = ReasoningContext(
    outer_context=data,
    api=llm_api,  # LLMAccessorAPI - creates LLMAccessorClient
    entrypoint_key="my_entrypoint"
)
```

## Example Usage

See the [example.py](example.py) file for a complete end-to-end demonstration with:

- **🔍 RAG-like Context Extraction**: Automatic context extraction from input data
- **🤖 Automatic LLM Client Detection**: Smart detection of API types with automatic client creation
- **🎛️ System Prompt Support**: Domain expertise and consistent persona across all reasoning steps
- **🔗 Direct mmar-llm Integration**: Seamless EntrypointsAccessor and LLMAccessorAPI usage
- **🌍 Multi-language Support**: Russian/English with easy extensibility
- **⚡ Parallel Execution**: DAG-based parallel processing
- **⚙️ Error Handling**: Comprehensive retry logic and error management
- **📊 Performance Metrics**: Execution timing and statistics

Run it with:

```bash
# Set entrypoints configuration
export ENTRYPOINTS_PATH=/path/to/your/entrypoints.json

# Run the demonstration
python example.py entrypoints.json my_entrypoint_key

# Or run with environment variable
ENTRYPOINTS_PATH=entrypoints.json python example.py
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
