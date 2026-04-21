#!/usr/bin/env python3
"""
Test script to verify CARL works with minimal dependencies (no vector-search).

Note: This is a standalone script, not a pytest test. Run it directly:
    python test_installation.py
"""

# Prevent pytest from collecting this as a test module
__test__ = False

import sys

print("Testing CARL with minimal dependencies...")
print("=" * 60)

# Test 1: Import core modules
print("\n1. Testing core imports...")
try:
    from mmar_carl import (
        LLMStepDescription,
        ReasoningChain,
    )
    print("✓ Core imports successful")
except Exception as e:
    print(f"✗ Core imports failed: {e}")
    sys.exit(1)

# Test 2: Substring search (no extra deps)
print("\n2. Testing substring search...")
try:
    from mmar_carl.models.search import ContextSearchConfig

    config = ContextSearchConfig(strategy="substring")
    strategy = config.get_strategy()

    context = "Revenue: $2.5M\nProfit: $700K\nEBITDA: 32%"
    queries = ["Revenue", "Profit"]
    result = strategy.extract_context(context, queries)

    assert "Revenue" in result or "Profit" in result
    print("✓ Substring search works")
except Exception as e:
    print(f"✗ Substring search failed: {e}")
    sys.exit(1)

# Test 3: Vector search should fail gracefully
print("\n3. Testing vector search error handling...")
try:
    from mmar_carl.models.search import VectorSearchStrategy

    try:
        strategy = VectorSearchStrategy()
        print("✗ Vector search should have raised ImportError")
        sys.exit(1)
    except ImportError as e:
        if "pip install 'mmar-carl[vector-search]'" in str(e):
            print("✓ Vector search raises helpful error")
        else:
            print(f"✗ Vector search error message unclear: {e}")
            sys.exit(1)
except Exception as e:
    print(f"✗ Vector search test failed: {e}")
    sys.exit(1)

# Test 4: Optional deps module
print("\n4. Testing optional deps module...")
try:
    from mmar_carl._optional_deps import (
        check_vector_search_available,
        check_openai_available,
        require,
    )

    # Should return False (deps not installed)
    assert check_vector_search_available() is False
    assert isinstance(check_openai_available(), bool)

    # Should raise helpful error
    try:
        require("faiss")
        print("✗ require() should have raised ImportError")
        sys.exit(1)
    except ImportError as e:
        if "pip install" in str(e):
            print("✓ Optional deps module works correctly")
        else:
            print(f"✗ require() error unclear: {e}")
            sys.exit(1)
except Exception as e:
    print(f"✗ Optional deps module test failed: {e}")
    sys.exit(1)

# Test 5: Chain creation (basic)
print("\n5. Testing chain creation...")
try:
    steps = [
        LLMStepDescription(
            number=1,
            title="Test Step",
            aim="Test step creation",
            reasoning_questions="Does this work?",
            stage_action="Verify functionality",
            example_reasoning="This is a test",
        )
    ]

    chain = ReasoningChain(steps=steps, max_workers=1, enable_progress=False)
    print("✓ Chain creation works")
except Exception as e:
    print(f"✗ Chain creation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("\nCore CARL functionality works without vector-search dependencies.")
print("Install vector-search with: pip install 'mmar-carl[vector-search]'")
