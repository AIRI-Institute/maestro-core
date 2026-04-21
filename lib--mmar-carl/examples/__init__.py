"""
CARL Examples Module

This module contains example scripts demonstrating various CARL features:

- basic_chain_example.py: Fundamental concepts of chain creation and execution
- openrouter_example.py: Using CARL with OpenRouter and OpenAI-compatible APIs
- tool_steps_example.py: Tool steps, memory operations, and mixed chains
- conditions_example.py: Conditional branching and routing in chains
- structured_output_example.py: Schema-constrained JSON output via Pydantic models
- llm_council_example.py: LLM Council — multiple models voting on a decision
- reflection_example.py: Reflection feature for analyzing chain execution results
- execution_modes_pipeline_example.py: Real API demo of FAST + SELF_CRITIC production flow
- execution_modes_mock_example.py: Mocked local demo of FAST + SELF_CRITIC production flow
- replan_deterministic_example.py: Rule-based RE-PLAN retry flow
- replan_llm_checker_example.py: LLM-based RE-PLAN checker with structured verdict
- replan_checkpoint_rollback_example.py: Named checkpoint rollback behavior
- replan_voting_example.py: Multi-checker voting and aggregation semantics
- replan_budget_guard_example.py: Budget safeguards and loop-prevention failure

Run examples directly:
    python examples/basic_chain_example.py
    python examples/openrouter_example.py
    python examples/tool_steps_example.py
    python examples/conditions_example.py
    python examples/structured_output_example.py
    python examples/llm_council_example.py
    python examples/reflection_example.py
    python examples/execution_modes_pipeline_example.py
    python examples/execution_modes_mock_example.py
    python examples/replan_deterministic_example.py
    python examples/replan_llm_checker_example.py
    python examples/replan_checkpoint_rollback_example.py
    python examples/replan_voting_example.py
    python examples/replan_budget_guard_example.py

Or import specific utilities:
    from examples.tool_steps_example import calculate_statistics, calculate_growth_rate
    from examples.utils import format_status
"""

from examples.utils import format_status

__all__ = [
    "basic_chain_example",
    "openrouter_example",
    "tool_steps_example",
    "conditions_example",
    "structured_output_example",
    "llm_council_example",
    "reflection_example",
    "execution_modes_pipeline_example",
    "execution_modes_mock_example",
    "replan_deterministic_example",
    "replan_llm_checker_example",
    "replan_checkpoint_rollback_example",
    "replan_voting_example",
    "replan_budget_guard_example",
    "utils",
    "format_status",
]
