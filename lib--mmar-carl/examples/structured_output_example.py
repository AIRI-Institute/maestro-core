#!/usr/bin/env python3
"""
Example: Structured Output Steps in CARL.

This example demonstrates how to use StructuredOutputStepDescription to
constrain LLM output to a specific JSON schema. This is useful when you
need predictable, machine-readable results from reasoning chains.

Three approaches are shown:
1. Direct JSON schema definition
2. Schema derived from a Pydantic model (recommended)
3. Multi-step chain: LLM reasoning → structured output extraction

Prerequisites:
    pip install 'mmar-carl[openai]'
    # or: pip install openai

Environment variables:
    OPENAI_API_KEY: Your OpenRouter API key (https://openrouter.ai)
    # or any other OpenAI-compatible API via LOCAL_LLM_URL / LOCAL_LLM_MODEL

Usage:
    export OPENAI_API_KEY="sk-or-v1-..."
    python examples/structured_output_example.py
"""

import asyncio
import json
import os

from pydantic import BaseModel, Field

from mmar_carl import (
    Language,
    LLMStepConfig,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    StructuredOutputStepConfig,
    StructuredOutputStepDescription,
    create_openai_client,
)
from examples.utils import format_status


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

FINANCIAL_REPORT = """
Company: Acme Corp
Period: FY 2024

Revenue:
  Q1: $1,500,000  (+5.2% YoY)
  Q2: $1,650,000  (+10.0% YoY)
  Q3: $1,720,000  (+4.2% YoY)
  Q4: $1,850,000  (+7.6% YoY)
  Full year: $6,720,000

Expenses:
  Q1: $1,200,000
  Q2: $1,280,000
  Q3: $1,350,000
  Q4: $1,400,000
  Full year: $5,230,000

Key highlights:
- Gross margin improved from 20% (Q1) to 24.3% (Q4)
- R&D investment up 15% compared to FY 2023
- Headcount grew from 120 to 145 employees
- Launched 2 new product lines in H2
"""


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------


class QuarterlyMetric(BaseModel):
    quarter: str = Field(..., description="Quarter label, e.g. 'Q1'")
    revenue: float = Field(..., description="Revenue in USD")
    expenses: float = Field(..., description="Expenses in USD")
    profit: float = Field(..., description="Profit in USD (revenue - expenses)")
    profit_margin_pct: float = Field(..., description="Profit margin as a percentage")


class FinancialSummary(BaseModel):
    company: str = Field(..., description="Company name")
    fiscal_year: int = Field(..., description="Fiscal year")
    total_revenue: float = Field(..., description="Full-year revenue in USD")
    total_expenses: float = Field(..., description="Full-year expenses in USD")
    total_profit: float = Field(..., description="Full-year profit in USD")
    average_profit_margin_pct: float = Field(..., description="Average profit margin across all quarters")
    quarterly_breakdown: list[QuarterlyMetric] = Field(..., description="Per-quarter metrics")
    key_highlights: list[str] = Field(..., description="Up to 5 key business highlights")
    outlook: str = Field(..., description="One-sentence outlook for next year")


class RiskAssessment(BaseModel):
    risk_level: str = Field(..., description="Overall risk level: low / medium / high")
    main_risks: list[str] = Field(..., description="Top 3 identified risk factors")
    confidence_score: float = Field(..., description="Analyst confidence score 0.0–1.0")
    recommendation: str = Field(..., description="Buy / Hold / Sell recommendation")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_client():
    """Create an LLM client from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return create_openai_client(
            api_key=api_key,
            model=os.environ.get("DEFAULT_EXAMPLES_MODEL", "Sber/GigaChat-Max-V2"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,  # Low temperature for structured output
        )

    # Fall back to a local LLM if available
    local_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")
    local_model = os.environ.get("LOCAL_LLM_MODEL", "llama3")
    return create_openai_client(
        api_key="not-needed",
        model=local_model,
        base_url=local_url,
        temperature=0.2,
        timeout=300.0,
    )


# ---------------------------------------------------------------------------
# Example 1 – Direct JSON schema
# ---------------------------------------------------------------------------


async def example_direct_schema():
    """
    Example 1: Structured output with a manually written JSON Schema.

    Use this approach when you want full control over the schema or when
    the output model is simple enough to define inline.
    """
    print("\n" + "=" * 60)
    print("Example 1: Direct JSON Schema")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LOCAL_LLM_URL")
    if not api_key:
        print("Skipping: set OPENAI_API_KEY or LOCAL_LLM_URL to run this example")
        return

    client = _make_client()

    # Define the output schema inline
    schema = {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "total_revenue": {"type": "number"},
            "total_profit": {"type": "number"},
            "profit_margin_pct": {"type": "number"},
            "highlights": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
            },
        },
        "required": ["company", "total_revenue", "total_profit", "profit_margin_pct", "highlights"],
    }

    steps = [
        StructuredOutputStepDescription(
            number=1,
            title="Extract Financial Summary",
            config=StructuredOutputStepConfig(
                input_source="$outer_context",
                output_schema=schema,
                schema_name="FinancialSnapshot",
                instruction="Extract the key financial figures from the annual report.",
                strict_json=True,
            ),
        ),
    ]

    chain = ReasoningChain(steps=steps, max_workers=1, trace_name="Direct JSON Schema")
    context = ReasoningContext(
        outer_context=FINANCIAL_REPORT,
        api=client,
        model="unused",
        language=Language.ENGLISH,
    )

    result = await chain.execute_async(context)

    print(f"\nExecution: {format_status(result.success)}")
    if result.success:
        step_result = result.step_results[0]
        print("\nRaw JSON string:")
        print(step_result.result)
        print("\nParsed Python object (result_data):")
        print(json.dumps(step_result.result_data, indent=2, ensure_ascii=False))
    else:
        for sr in result.step_results:
            if not sr.success:
                print(f"  Step {sr.step_number} error: {sr.error_message}")


# ---------------------------------------------------------------------------
# Example 2 – Schema from Pydantic model (recommended)
# ---------------------------------------------------------------------------


async def example_pydantic_schema():
    """
    Example 2: Structured output derived from a Pydantic model.

    StructuredOutputStepConfig.from_pydantic_model() converts the model's
    JSON schema automatically. The result_data dict can then be validated
    back into the Pydantic model with model_validate().
    """
    print("\n" + "=" * 60)
    print("Example 2: Schema from Pydantic Model (Recommended)")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LOCAL_LLM_URL")
    if not api_key:
        print("Skipping: set OPENAI_API_KEY or LOCAL_LLM_URL to run this example")
        return

    client = _make_client()

    steps = [
        StructuredOutputStepDescription(
            number=1,
            title="Parse Financial Report",
            config=StructuredOutputStepConfig.from_pydantic_model(
                FinancialSummary,
                input_source="$outer_context",
                instruction="Parse the annual report and populate all fields precisely.",
                strict_json=True,
            ),
        ),
    ]

    chain = ReasoningChain(steps=steps, max_workers=1, trace_name="Pydantic Schema")
    context = ReasoningContext(
        outer_context=FINANCIAL_REPORT,
        api=client,
        model="unused",
        language=Language.ENGLISH,
    )

    result = await chain.execute_async(context)

    print(f"\nExecution: {format_status(result.success)}")
    if result.success:
        step_result = result.step_results[0]

        # Validate the parsed dict back into the Pydantic model
        try:
            summary = FinancialSummary.model_validate(step_result.result_data)
            print(f"\nCompany      : {summary.company}")
            print(f"Fiscal Year  : {summary.fiscal_year}")
            print(f"Total Revenue: ${summary.total_revenue:,.0f}")
            print(f"Total Profit : ${summary.total_profit:,.0f}")
            print(f"Avg Margin   : {summary.average_profit_margin_pct:.1f}%")
            print(f"Outlook      : {summary.outlook}")
            print("\nQuarterly Breakdown:")
            for q in summary.quarterly_breakdown:
                print(f"  {q.quarter}: revenue=${q.revenue:,.0f}  margin={q.profit_margin_pct:.1f}%")
            print("\nKey Highlights:")
            for h in summary.key_highlights:
                print(f"  - {h}")
        except Exception as e:
            print(f"Model validation failed (raw dict follows): {e}")
            print(json.dumps(step_result.result_data, indent=2, ensure_ascii=False))
    else:
        for sr in result.step_results:
            if not sr.success:
                print(f"  Step {sr.step_number} error: {sr.error_message}")


# ---------------------------------------------------------------------------
# Example 3 – Multi-step: LLM reasoning → structured extraction
# ---------------------------------------------------------------------------


async def example_multistep_chain():
    """
    Example 3: Combining LLM reasoning with structured output extraction.

    A common pattern:
      Step 1 (LLM)              – free-form reasoning / analysis
      Step 2 (LLM, optional)    – deeper analysis building on step 1
      Step 3 (STRUCTURED_OUTPUT) – convert accumulated reasoning into a
                                    structured object for downstream use

    The structured output step reads $history[-1] by default, meaning it
    receives the last step's text output as input.
    """
    print("\n" + "=" * 60)
    print("Example 3: Multi-step – LLM Reasoning + Structured Extraction")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LOCAL_LLM_URL")
    if not api_key:
        print("Skipping: set OPENAI_API_KEY or LOCAL_LLM_URL to run this example")
        return

    client = _make_client()

    steps = [
        # Step 1: Free-form financial analysis
        LLMStepDescription(
            number=1,
            title="Financial Analysis",
            aim="Analyse the company's financial performance and identify key risk factors",
            reasoning_questions=(
                "How has profitability evolved over the year? "
                "What risks could threaten future performance? "
                "What is the overall investment recommendation?"
            ),
            stage_action="Review all quarters, compute margins, list risks, form a recommendation",
            example_reasoning=(
                "Margins improved from 20% to 24.3%, indicating operational efficiency gains. "
                "Main risks: revenue concentration, rising headcount costs. Recommendation: Buy."
            ),
            step_context_queries=["revenue", "expenses", "profit", "margin", "risk"],
            llm_config=LLMStepConfig(temperature=0.5),
        ),
        # Step 2: Convert the free-form analysis into a typed RiskAssessment object
        StructuredOutputStepDescription(
            number=2,
            title="Structure Risk Assessment",
            dependencies=[1],  # waits for step 1 to finish
            config=StructuredOutputStepConfig.from_pydantic_model(
                RiskAssessment,
                input_source="$history[-1]",  # uses the output of step 1
                instruction=(
                    "Based on the analysis above, produce a structured risk assessment. "
                    "confidence_score must be between 0.0 and 1.0."
                ),
                strict_json=True,
            ),
        ),
    ]

    chain = ReasoningChain(steps=steps, max_workers=2, trace_name="Multi-step LLM + Structured Output")
    context = ReasoningContext(
        outer_context=FINANCIAL_REPORT,
        api=client,
        model="unused",
        language=Language.ENGLISH,
        system_prompt="You are a senior investment analyst. Be precise and data-driven.",
    )

    print("Executing 2-step chain (LLM reasoning → structured extraction)...")
    result = await chain.execute_async(context)

    print(f"\nExecution: {format_status(result.success)}")
    print(f"Total time: {result.total_execution_time:.2f}s")

    if result.success:
        # Step 1 – free-form text
        analysis_result = result.step_results[0]
        print("\n--- Step 1: Free-form Analysis (excerpt) ---")
        excerpt = analysis_result.result[:400].rstrip()
        print(excerpt + ("..." if len(analysis_result.result) > 400 else ""))

        # Step 2 – structured JSON
        struct_result = result.step_results[1]
        print("\n--- Step 2: Structured Risk Assessment ---")
        try:
            assessment = RiskAssessment.model_validate(struct_result.result_data)
            print(f"Risk Level     : {assessment.risk_level.upper()}")
            print(f"Recommendation : {assessment.recommendation}")
            print(f"Confidence     : {assessment.confidence_score:.0%}")
            print("Main Risks:")
            for risk in assessment.main_risks:
                print(f"  - {risk}")
        except Exception as e:
            print(f"Validation failed: {e}")
            print(json.dumps(struct_result.result_data, indent=2, ensure_ascii=False))
    else:
        for sr in result.step_results:
            if not sr.success:
                print(f"  Step {sr.step_number} ({sr.step_title}) error: {sr.error_message}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    print("CARL – Structured Output Step Examples")
    print("=" * 60)
    print(
        "\nThese examples show how to use StructuredOutputStepDescription to\n"
        "constrain LLM responses to a specific JSON schema.\n"
        "\nSet OPENAI_API_KEY (or LOCAL_LLM_URL + LOCAL_LLM_MODEL) to run."
    )

    await example_direct_schema()
    await example_pydantic_schema()
    await example_multistep_chain()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
