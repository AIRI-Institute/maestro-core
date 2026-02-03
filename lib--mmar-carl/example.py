#!/usr/bin/env python3
"""
End-to-end example of using CARL (Collaborative Agent Reasoning Library).

This example demonstrates:
1. Creating reasoning steps with dependencies
2. Setting up mmar-llm integration
3. Using both Russian and English languages
4. Executing reasoning chains with parallel processing
5. Getting detailed results and metrics
"""

import asyncio
import json
import os
import traceback
import sys

from src.mmar_carl import (
    ContextQuery,
    ContextSearchConfig,
    Language,
    ReasoningChain,
    ReasoningContext,
    StepDescription,
)
from mmar_llm import LLMConfig, LLMHub
from mmar_ptag import ptag_client

# Example financial data (CSV format)
SAMPLE_DATA = """Период,EBITDA,SALES_REVENUE,NET_INCOME,OPERATING_CASH_FLOW
2023-Q1,1000000,5000000,800000,950000
2023-Q2,1200000,5500000,950000,1100000
2023-Q3,1100000,5200000,850000,1000000
2023-Q4,1300000,5800000,1050000,1200000
2024-Q1,1400000,6000000,1150000,1250000
2024-Q2,1350000,5900000,1100000,1180000"""

# EBITDA Analysis Chain - Simple sequential example
EBITDA_ANALYSIS = [
    StepDescription(
        number=1,
        title="Рост EBITDA и стабильность",
        aim="Выяснить, стабильно ли растет EBITDA и насколько выросла",
        reasoning_questions="Насколько изменилась (выросла/уменьшилась) EBITDA за рассматриваемый период (год-к-году)?",
        dependencies=[],
        step_context_queries=["EBITDA"],
        stage_action="Рассчитать темп прироста EBITDA за 12 последних месяцев год к году",
        example_reasoning="1) если темп > 0%, то положительный сигнал; если темп < 0%, то отрицательный сигнал; 2) если темп за период расчета больше, чем за предыдущий аналогичный период, то положительный сигнал",
    ),
    StepDescription(
        number=2,
        title="Маржинальность EBITDA",
        aim="Выяснить, сколько процентов выручки забирают операционные расходы (без амортизации) и насколько эта ситуация стабильна",
        reasoning_questions="Какая маржинальность по EBITDA за рассматриваемый период (за последние 12 месяцев)? Идет рост/уменьшение маржи по EBITDA?",
        dependencies=[1],
        step_context_queries=["EBITDA", "SALES_REVENUE"],
        stage_action="EBITDA маржа = EBITDA / SALES_REVENUE",
        example_reasoning="1) если маржа > 0%, то положительный сигнал: компания зарабатывает больше своих операционных расходов; 2) если маржа выросла по сравнению с предыдущим периодом, то положительный сигнал",
    ),
]

# Advanced Search Configuration Example - Demonstrates per-query search strategies
ADVANCED_SEARCH_EXAMPLE = [
    StepDescription(
        number=1,
        title="Comprehensive Financial Analysis",
        aim="Analyze financial metrics using mixed search strategies",
        reasoning_questions="What are the key financial indicators and their trends?",
        dependencies=[],
        # Mix of string queries and ContextQuery objects with different strategies
        step_context_queries=[
            "EBITDA",  # Simple string query (uses chain default)
            ContextQuery(
                query="SALES_REVENUE",
                search_strategy="vector",
                search_config={"similarity_threshold": 0.8, "max_results": 3},
            ),
            ContextQuery(
                query="NET_INCOME",
                search_strategy="substring",
                search_config={"case_sensitive": True, "min_word_length": 4},
            ),
        ],
        stage_action="Extract comprehensive financial insights",
        example_reasoning="Mixed search strategies provide both semantic and exact matching for comprehensive analysis",
    ),
    StepDescription(
        number=2,
        title="Trend Analysis with Vector Search",
        aim="Identify trends using semantic search",
        reasoning_questions="What trends and patterns emerge from the data?",
        dependencies=[1],
        step_context_queries=[
            ContextQuery(
                query="growth trends financial performance",
                search_strategy="vector",
                search_config={
                    "embedding_model": "all-MiniLM-L6-v2",
                    "similarity_threshold": 0.7,
                    "index_type": "flat",
                },
            ),
            ContextQuery(
                query="quarterly patterns",
                search_strategy="vector",
                search_config={
                    "similarity_threshold": 0.9,  # Higher threshold for more specific matches
                    "max_results": 2,
                },
            ),
        ],
        stage_action="Analyze trends using semantic similarity",
        example_reasoning="Vector search captures semantic relationships beyond exact text matching",
    ),
]

# Parallel Analysis Chain - Demonstrates parallel execution
PARALLEL_ANALYSIS = [
    StepDescription(
        number=1,
        title="Анализ выручки",
        aim="Оценить динамику и структуру выручки",
        reasoning_questions="Как изменилась выручка за период? Какие сегменты растут/снижаются?",
        dependencies=[],
        step_context_queries=["SALES_REVENUE"],
        stage_action="Проанализировать динамику выручки по сегментам",
        example_reasoning="Рост выручки является положительным сигналом, особенно если он опережает инфляцию и рост рынка",
    ),
    StepDescription(
        number=2,
        title="Анализ операционных расходов",
        aim="Оценить эффективность управления операционными расходами",
        reasoning_questions="Как изменились операционные расходы? Соотношение расходов к выручке?",
        dependencies=[],
        step_context_queries=["OPERATING_CASH_FLOW", "NET_INCOME"],
        stage_action="Проанализировать структуру и динамику операционных расходов",
        example_reasoning="Снижение операционных расходов в % от выручки указывает на повышение эффективности",
    ),
    StepDescription(
        number=3,
        title="Анализ маржинальности",
        aim="Оценить рентабельность операционной деятельности",
        reasoning_questions="Какая операционная маржа? Как она меняется?",
        dependencies=[1, 2],  # Depends on both revenue and expense analysis
        step_context_queries=[],
        stage_action="Рассчитать операционную маржу и сравнить с предыдущими периодами",
        example_reasoning="Рост операционной маржи говорит об улучшении операционной эффективности",
    ),
]

# English version of the chain for comparison
ENGLISH_EBITDA_ANALYSIS = [
    StepDescription(
        number=1,
        title="EBITDA Growth Analysis",
        aim="Analyze EBITDA growth and stability",
        reasoning_questions="How has EBITDA changed over the period (year-over-year)?",
        dependencies=[],
        step_context_queries=["EBITDA"],
        stage_action="Calculate EBITDA growth rate for the last 12 months year-over-year",
        example_reasoning="1) If growth > 0%, it's a positive signal; if growth < 0%, it's a negative signal; 2) If growth rate for the period is higher than the previous similar period, it's a positive signal",
    ),
    StepDescription(
        number=2,
        title="EBITDA Margin Analysis",
        aim="Analyze what percentage of revenue is taken by operating expenses (excluding depreciation) and how stable this situation is",
        reasoning_questions="What is the EBITDA margin for the period (last 12 months)? Is the EBITDA margin growing/declining?",
        dependencies=[1],
        step_context_queries=["EBITDA"],
        stage_action="EBITDA margin = EBITDA / SALES_REVENUE",
        example_reasoning="1) If margin > 0%, it's a positive signal: the company earns more than its operating expenses; 2) If margin increased compared to the previous period, it's a positive signal",
    ),
]


def create_api(entrypoints_path: str | None = None):
    return create_llm_accessor()  # or create_entrypoints(entrypoints_path=entrypoints_path)


def create_entrypoints(entrypoints_path: str | None = None):
    if entrypoints_path is None:
        # Try to find entrypoints configuration in common locations
        possible_paths = [
            os.environ.get("ENTRYPOINTS_PATH"),
            "entrypoints.json",
            "config/entrypoints.json",
            "../entrypoints.json",
            "../../config/entrypoints.json",
        ]

        for path in possible_paths:
            if path and os.path.exists(path):
                entrypoints_path = path
                break
        else:
            raise FileNotFoundError(
                "Entrypoints configuration not found. "
                "Set ENTRYPOINTS_PATH environment variable or provide entrypoints.json file."
            )

    if entrypoints_path and not os.path.exists(entrypoints_path):
        raise FileNotFoundError(f"Entrypoints configuration file not found: {entrypoints_path}")
    if not entrypoints_path:
        raise FileNotFoundError("Entrypoints configuration file path is empty")

    try:
        # Load entrypoints from JSON configuration
        # LLMHub expects a config object, not a path
        with open(entrypoints_path, encoding="utf-8") as f:
            config_data = json.load(f)

        # Create proper LLMConfig
        entrypoints_config = LLMConfig.model_validate(config_data)
        return LLMHub(entrypoints_config)
    except Exception as e:
        raise ValueError(f"Failed to create entrypoints from {entrypoints_path}: {e}")


def create_llm_accessor():
    return ptag_client(LLMHub, 40631)


async def demonstrate_russian_chain(entrypoints_path: str | None = None, endpoint_key: str = "default"):
    """Demonstrate CARL usage with Russian language."""
    print("\n" + "=" * 60)
    print("🇷🇺 RUSSIAN LANGUAGE CHAIN DEMONSTRATION")
    print("=" * 60)

    # Create entrypoints accessor
    api = create_api(entrypoints_path)

    try:
        # System prompt for financial analysis expertise
        russian_system_prompt = """
Вы старший финансовый аналитик с многолетним опытом в анализе EBITDA и финансовых показателей.

Ваш анализ должен:
- Основываться на данных и доказательствах
- Включать конкретные проценты и числовые значения
- Предоставлять actionable выводы и рекомендации
- Рассматривать тренды и сезонные колебания
- Учитывать контекст рыночной ситуации и макроэкономических факторов
"""

        # Create reasoning context with Russian language (LLM client created automatically)
        context = ReasoningContext(
            outer_context=SAMPLE_DATA,
            api=api,
            endpoint_key=endpoint_key,
            retry_max=2,
            language=Language.RUSSIAN,
            system_prompt=russian_system_prompt.strip(),
        )

        # Create reasoning chain
        chain = ReasoningChain(
            steps=EBITDA_ANALYSIS,
            max_workers=2,
            enable_progress=True,
        )

        print("📊 Input Data:")
        print(SAMPLE_DATA[:100] + "...")
        print(f"\n🔗 Execution Plan: {chain.get_execution_plan()}")
        print(f"📋 Steps Summary: {len(chain.steps)} steps")
        print("🌐 Language: Russian")

        # Execute the chain
        print("\n⚡ Executing reasoning chain...")
        result = chain.execute(context)

        # Display results
        print(f"\n✅ Execution Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"⏱️  Total Execution Time: {result.total_execution_time:.2f}s")
        print(f"📈 Successful Steps: {len(result.get_successful_steps())}")
        print(f"❌ Failed Steps: {len(result.get_failed_steps())}")

        if result.success:
            print("\n📝 Reasoning Output:")
            print("-" * 40)
            print(result.get_final_output())
            print("-" * 40)

        # Show detailed step results
        print("\n📋 Detailed Step Results:")
        for step_result in result.step_results:
            status = "✅" if step_result.success else "❌"
            time_str = f"{step_result.execution_time:.2f}s" if step_result.execution_time else "N/A"
            print(f"{status} Step {step_result.step_number}: {step_result.step_title} ({time_str})")
            if not step_result.success:
                print(f"   Error: {step_result.error_message}")

        # Show execution metrics
        metadata = result.metadata.get("execution_stats", {})
        if metadata:
            print("\n📊 Execution Statistics:")
            print(f"   Parallel Batches: {metadata.get('parallel_batches', 'N/A')}")
            print(f"   Total Steps: {metadata.get('total_steps', 'N/A')}")
            print(f"   Executed Steps: {metadata.get('executed_steps', 'N/A')}")

    except Exception as e:
        print(f"\n❌ Russian chain demonstration failed: {e}")
        raise

    # No cleanup needed - entrypoint key is handled cleanly


async def demonstrate_english_chain(entrypoints_path: str | None = None, endpoint_key: str = "default"):
    """Demonstrate CARL usage with English language."""
    print("\n" + "=" * 60)
    print("🇺🇸 ENGLISH LANGUAGE CHAIN DEMONSTRATION")
    print("=" * 60)

    # Create entrypoints accessor
    api = create_api(entrypoints_path)

    try:
        # System prompt for financial analysis expertise
        english_system_prompt = """
You are a senior financial analyst with extensive experience in EBITDA analysis and financial metrics evaluation.

Your analysis should:
- Be data-driven and evidence-based
- Include specific percentages and numerical values
- Provide actionable insights and recommendations
- Consider trends and seasonal fluctuations
- Account for market context and macroeconomic factors
- Maintain professional objectivity and analytical rigor
"""

        # Create reasoning context with English language (LLM client created automatically)
        context = ReasoningContext(
            outer_context=SAMPLE_DATA,
            api=api,
            endpoint_key=endpoint_key,
            retry_max=2,
            language=Language.ENGLISH,
            system_prompt=english_system_prompt.strip(),
        )

        # Create reasoning chain with English steps
        chain = ReasoningChain(
            steps=ENGLISH_EBITDA_ANALYSIS,
            max_workers=2,
            enable_progress=True,
        )

        print("📊 Input Data:")
        print(SAMPLE_DATA[:100] + "...")
        print(f"\n🔗 Execution Plan: {chain.get_execution_plan()}")
        print(f"📋 Steps Summary: {len(chain.steps)} steps")
        print("🌐 Language: English")

        # Execute the chain
        print("\n⚡ Executing reasoning chain...")
        result = chain.execute(context)

        # Display results
        print(f"\n✅ Execution Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"⏱️  Total Execution Time: {result.total_execution_time:.2f}s")
        print(f"📈 Successful Steps: {len(result.get_successful_steps())}")
        print(f"❌ Failed Steps: {len(result.get_failed_steps())}")

        if result.success:
            print("\n📝 Reasoning Output:")
            print("-" * 40)
            print(result.get_final_output())
            print("-" * 40)

    except Exception as e:
        print(f"\n❌ English chain demonstration failed: {e}")
        raise


async def demonstrate_parallel_execution(entrypoints_path: str | None = None, endpoint_key: str = "default"):
    """Demonstrate parallel execution capabilities."""
    print("\n" + "=" * 60)
    print("🚀 PARALLEL EXECUTION DEMONSTRATION")
    print("=" * 60)

    # Create entrypoints accessor
    api = create_api(entrypoints_path)

    try:
        # System prompt for comprehensive financial analysis
        parallel_system_prompt = """
Вы руководитель отдела финансового анализа с командой аналитиков.

Ваша оценка должна быть:
- Комплексной и всесторонней, охватывающей все аспекты финансовой деятельности
- Основанной на количественных данных и качественных показателях
- Сфокусированной на выявлении возможностей для оптимизации
- Практической и ориентированной на принятие управленческих решений
- Учитывающей как краткосрочные, так и долгосрочные перспективы
"""

        # Create reasoning context (LLM client created automatically)
        context = ReasoningContext(
            outer_context=SAMPLE_DATA,
            api=api,
            endpoint_key=endpoint_key,
            retry_max=2,
            language=Language.RUSSIAN,
            system_prompt=parallel_system_prompt.strip(),
        )

        # Create parallel reasoning chain
        chain = ReasoningChain(
            steps=PARALLEL_ANALYSIS,
            max_workers=3,  # Allow up to 3 parallel executions
            enable_progress=True,
        )

        print("📊 Input Data:")
        print(SAMPLE_DATA[:100] + "...")
        print(f"\n🔗 Execution Plan: {chain.get_execution_plan()}")
        print(f"📋 Steps Summary: {len(chain.steps)} steps")
        print(f"🚀 Max Workers: {chain.max_workers}")

        # Show dependency analysis
        dependencies = chain.get_step_dependencies()
        print("\n🔗 Step Dependencies:")
        for step_num, deps in dependencies.items():
            if deps:
                print(f"   Step {step_num} depends on: {deps}")
            else:
                print(f"   Step {step_num} has no dependencies (can run in parallel)")

        # Execute the chain
        print("\n⚡ Executing parallel reasoning chain...")
        result = chain.execute(context)

        # Display results
        print(f"\n✅ Execution Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"⏱️  Total Execution Time: {result.total_execution_time:.2f}s")
        print(f"📈 Successful Steps: {len(result.get_successful_steps())}")
        print(f"❌ Failed Steps: {len(result.get_failed_steps())}")

        if result.success:
            print("\n📝 Reasoning Output:")
            print("-" * 40)
            print(result.get_final_output())
            print("-" * 40)

        # Show parallel execution benefits
        metadata = result.metadata.get("execution_stats", {})
        parallel_batches = metadata.get("parallel_batches", 0)
        print("\n🚀 Parallel Execution Benefits:")
        print(f"   Parallel Batches: {parallel_batches}")
        if parallel_batches > 1:
            print("   ✅ Steps were executed in parallel batches")
            print("   ⚡ Improved performance through parallelization")
        else:
            print("   ℹ️  Sequential execution (steps had dependencies)")

    except Exception as e:
        print(f"\n❌ Parallel execution demonstration failed: {e}")
        raise


async def demonstrate_advanced_search(entrypoints_path: str | None = None, endpoint_key: str = "default"):
    """Demonstrate advanced per-query search configuration."""
    print("\n" + "=" * 60)
    print("🔍 ADVANCED SEARCH CONFIGURATION DEMONSTRATION")
    print("=" * 60)

    # Create entrypoints accessor
    api = create_api(entrypoints_path)

    try:
        # System prompt for expert financial analysis with search capabilities
        search_system_prompt = """
You are an expert financial analyst with advanced analytical skills and deep understanding of both traditional and modern financial analysis techniques.

Your analysis should:
- Utilize multiple search strategies to extract comprehensive insights
- Combine quantitative analysis with qualitative assessment
- Leverage semantic search for trend identification and pattern recognition
- Apply substring search for precise metric extraction
- Provide nuanced recommendations based on multi-dimensional analysis
- Consider both historical performance and forward-looking indicators
- Maintain analytical rigor while adapting insights to business context
"""

        # Create reasoning context (LLM client created automatically)
        context = ReasoningContext(
            outer_context=SAMPLE_DATA,
            api=api,
            endpoint_key=endpoint_key,
            retry_max=2,
            language=Language.ENGLISH,
            system_prompt=search_system_prompt.strip(),
        )

        # Create chain with default substring search
        chain = ReasoningChain(
            steps=ADVANCED_SEARCH_EXAMPLE,
            search_config=ContextSearchConfig(
                strategy="substring",  # Default strategy
                substring_config={"case_sensitive": False, "min_word_length": 2, "max_matches_per_query": 3},
            ),
            max_workers=2,
            enable_progress=True,
        )

        print("📊 Input Data:")
        print(SAMPLE_DATA[:100] + "...")
        print(f"\n🔗 Chain Default Search Strategy: {chain.prompt_template.search_config.strategy}")
        print("📋 Steps with Mixed Search Strategies:")

        for step in chain.steps:
            print(f"   Step {step.number}: {step.title}")
            for i, query in enumerate(step.step_context_queries):
                if isinstance(query, str):
                    print(f"     Query {i + 1}: '{query}' (uses chain default)")
                else:  # ContextQuery object
                    print(f"     Query {i + 1}: '{query.query}' (strategy: {query.search_strategy})")
                    if query.search_config:
                        print(f"       Config: {query.search_config}")

        # Execute the chain
        print("\n⚡ Executing advanced search chain...")
        result = chain.execute(context)

        # Display results
        print(f"\n✅ Execution Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"⏱️  Total Execution Time: {result.total_execution_time:.2f}s")
        print(f"📈 Successful Steps: {len(result.get_successful_steps())}")

        if result.success:
            print("\n📝 Reasoning Output:")
            print("-" * 40)
            print(result.get_final_output())
            print("-" * 40)

        print("\n🔍 Advanced Search Features Demonstrated:")
        print("  ✅ Mixed search strategies within same step")
        print("  ✅ Per-query search strategy override")
        print("  ✅ Query-specific configuration parameters")
        print("  ✅ Fallback to chain default when no override")
        print("  ✅ Both vector and substring search in one chain")

    except Exception as e:
        print(f"\n❌ Advanced search demonstration failed: {e}")
        raise


async def main():
    """Main demonstration function."""
    print("🎯 MAESTRO CARL - Collaborative Agent Reasoning Library")
    print("🔬 End-to-End Demonstration with Advanced Search")
    print("=" * 60)

    # Get entrypoints configuration from command line arguments or environment
    entrypoints_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ENTRYPOINTS_PATH")
    endpoint_key = sys.argv[2] if len(sys.argv) > 2 else "default"

    if entrypoints_path:
        print(f"📁 Using entrypoints configuration: {entrypoints_path}")
        print(f"🔑 Using entrypoint key: {endpoint_key}")
    else:
        print("📁 Searching for entrypoints configuration in default locations...")
        print(f"🔑 Using default entrypoint key: {endpoint_key}")

    try:
        # Demonstrate different use cases
        await demonstrate_russian_chain(entrypoints_path, endpoint_key)
        await demonstrate_english_chain(entrypoints_path, endpoint_key)
        await demonstrate_parallel_execution(entrypoints_path, endpoint_key)
        await demonstrate_advanced_search(entrypoints_path, endpoint_key)

        print("\n" + "=" * 60)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📚 Key Features Demonstrated:")
        print("  ✅ Direct mmar-llm LLMHub integration")
        print("  ✅ Multi-language support (Russian/English)")
        print("  ✅ System prompt support for domain expertise")
        print("  ✅ Parallel execution with DAG optimization")
        print("  ✅ Dependency management")
        print("  ✅ Error handling and retry logic")
        print("  ✅ Execution metrics and monitoring")
        print("  ✅ Flexible data input (outer_context)")
        print("  🔍 Configurable search strategies (substring/vector)")
        print("  🔎 Per-query search strategy override")
        print("  🔧 Mixed search strategies within single step")
        print("\n🚀 Ready to integrate with your financial reasoning service!")
        print("\n📖 Usage:")
        print("  python example.py [entrypoints.json] [endpoint_key]")
        print("  ENTRYPOINTS_PATH=entrypoints.json python example.py")
        print("  ENTRYPOINTS_PATH=entrypoints.json endpoint_key=my_key python example.py")

    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        print("\n💡 Make sure to:")
        print("  1. Set ENTRYPOINTS_PATH environment variable")
        print("  2. Or provide entrypoints.json as command line argument")
        print("  3. Or place entrypoints.json in current directory")
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
