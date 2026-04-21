#!/usr/bin/env python3
"""
Example runner for CARL examples.

Runs all examples and generates a summary table with execution status.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT = 1000


# Error patterns to detect in output (even if exit code is 0)
# These are more specific patterns to avoid false positives
ERROR_PATTERNS = [
    "Traceback (most recent",
    "Task exception was never retrieved",
    "Task was destroyed but it is pending",
    "AttributeError:",
    "TypeError:",
    "ValueError:",
    "KeyError:",
    "ImportError:",
    "RuntimeError:",
    "Exception ignored in:",
    "Unhandled exception",
    "Error:",  # With colon to avoid matching "error" in strings
]

# Examples that should have more lenient error checking
# (e.g., examples that test error conditions as part of their normal operation)
LENIENT_CHECK_EXAMPLES = {
    "conditions_example.py",  # Tests conditions with "error" in test data
    "reflection_example.py",  # Tests error handling with expected RuntimeError
}


# ANSI colors for table output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


# Examples that don't require API keys
NO_API_KEY_EXAMPLES = {
    "conditions_example.py",
    "execution_modes_mock_example.py",
    "metrics_example.py",
    "dataset_evaluator_example.py",
    "replan_deterministic_example.py",
}


class ExampleResult:
    """Result of running a single example."""

    def __init__(self, name: str):
        self.name = name
        self.status = "PENDING"
        self.execution_time = 0.0
        self.details = ""
        self.skipped_reason = ""
        self.success: bool = False  # Whether the example succeeded


def detect_errors_in_output(stdout: str, stderr: str, example_name: str = "") -> tuple[bool, str]:
    """Detect error patterns in output even if exit code is 0.

    Args:
        stdout: Standard output from the process
        stderr: Standard error from the process
        example_name: Name of the example file for lenient checking

    Returns:
        (has_errors, error_details)
    """
    # For lenient examples, only check for severe errors
    if example_name in LENIENT_CHECK_EXAMPLES:
        severe_patterns = [
            "Traceback (most recent",
            "Task exception was never retrieved",
            "Unhandled exception",
        ]
        all_output = f"{stderr}\n{stdout}"
        for pattern in severe_patterns:
            if pattern in all_output:
                for line in all_output.split("\n"):
                    if pattern in line:
                        return True, line.strip()[:100]
        return False, ""

    # For normal examples, check all error patterns
    # Combine stderr and stdout for error detection
    all_output = f"{stderr}\n{stdout}"

    # Check for error patterns
    for pattern in ERROR_PATTERNS:
        if pattern in all_output:
            # Find the line with the error
            for line in all_output.split("\n"):
                if pattern in line:
                    return True, line.strip()[:100]

    return False, ""


def check_api_key(example_file: str) -> tuple[bool, str]:
    """Check if required API keys are present for an example.

    Args:
        example_file: Name of the example file

    Returns:
        (has_key, reason_if_missing)
    """
    if example_file in NO_API_KEY_EXAMPLES:
        return True, ""

    # Check for OPENAI_API_KEY
    if not os.environ.get("OPENAI_API_KEY"):
        return False, "Missing OPENAI_API_KEY"

    return True, ""


def run_example(example_file: str, examples_dir: Path) -> ExampleResult:
    """Run a single example and capture results.

    Args:
        example_file: Name of the example file
        examples_dir: Path to the examples directory

    Returns:
        ExampleResult with execution details
    """
    result = ExampleResult(example_file)

    # Check API keys first
    has_key, skip_reason = check_api_key(example_file)
    if not has_key:
        result.status = "SKIPPED"
        result.skipped_reason = skip_reason
        result.details = skip_reason
        return result

    # Build command using uv run
    example_path = examples_dir / example_file
    cmd = ["uv", "run", "python", str(example_path)]

    # Run example and capture output
    start_time = time.time()
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,  # 2 minute timeout
            cwd=str(examples_dir),
            env=os.environ.copy(),
        )
        execution_time = time.time() - start_time
        result.execution_time = execution_time

        # Check for errors in output even if exit code is 0
        has_errors, error_details = detect_errors_in_output(process.stdout, process.stderr, example_file)

        # Determine success based on exit code and error detection
        exit_success = process.returncode == 0

        if has_errors:
            # Errors detected in output even if exit code is 0
            result.status = "FAILED"
            result.details = error_details
        elif not exit_success:
            # Non-zero exit code
            result.status = "FAILED"
            # Extract details from stderr
            if process.stderr:
                first_line = process.stderr.strip().split("\n")[0]
                result.details = first_line[:100]
            else:
                result.details = f"Exit code {process.returncode}"
        else:
            # True success
            result.status = "SUCCESS"
            # Extract details from stdout for successful runs
            if process.stdout:
                # Try to find a useful detail from stdout
                lines = process.stdout.strip().split("\n")
                # Look for status or completion messages
                for line in lines[-10:]:  # Check last 10 lines
                    if "completed" in line.lower() or "done" in line.lower() or "steps" in line.lower():
                        result.details = line[:100]
                        break
                else:
                    result.details = "Completed successfully"
            else:
                result.details = "Completed successfully"

    except subprocess.TimeoutExpired:
        result.status = "TIMEOUT"
        result.execution_time = float(TIMEOUT)
        result.details = f"Execution exceeded {TIMEOUT}s timeout (check for hanging async tasks)"
    except Exception as e:
        result.status = "FAILED"
        result.execution_time = time.time() - start_time
        result.details = str(e)[:100]

    return result


def print_summary_table(results: list[ExampleResult], total_time: float):
    """Print a beautiful summary table of results.

    Args:
        results: List of ExampleResult objects
        total_time: Total execution time for all examples
    """
    # Calculate statistics
    success_count = sum(1 for r in results if r.status == "SUCCESS")
    failed_count = sum(1 for r in results if r.status == "FAILED")
    skipped_count = sum(1 for r in results if r.status == "SKIPPED")
    timeout_count = sum(1 for r in results if r.status == "TIMEOUT")

    # Table formatting
    col_widths = {"name": 37, "status": 10, "time": 10, "details": 30}

    def format_row(name: str, status: str, time_str: str, details: str) -> str:
        """Format a single table row."""
        return (
            f"│ {name.ljust(col_widths['name'])} │ "
            f"{status.ljust(col_widths['status'])} │ "
            f"{time_str.ljust(col_widths['time'])} │ "
            f"{details.ljust(col_widths['details'])} │"
        )

    def format_status(status: str) -> str:
        """Add color to status."""
        if status == "SUCCESS":
            return f"{Colors.GREEN}{status}{Colors.RESET}"
        elif status == "FAILED":
            return f"{Colors.RED}{status}{Colors.RESET}"
        elif status == "SKIPPED":
            return f"{Colors.YELLOW}{status}{Colors.RESET}"
        elif status == "TIMEOUT":
            return f"{Colors.BLUE}{status}{Colors.RESET}"
        return status

    # Print table
    separator = (
        "├"
        + "─" * (col_widths["name"] + 2)
        + "┼"
        + "─" * (col_widths["status"] + 2)
        + "┼"
        + "─" * (col_widths["time"] + 2)
        + "┼"
        + "─" * (col_widths["details"] + 2)
        + "┤"
    )
    top_border = (
        "┌"
        + "─" * (col_widths["name"] + 2)
        + "┬"
        + "─" * (col_widths["status"] + 2)
        + "┬"
        + "─" * (col_widths["time"] + 2)
        + "┬"
        + "─" * (col_widths["details"] + 2)
        + "┐"
    )
    bottom_border = (
        "└"
        + "─" * (col_widths["name"] + 2)
        + "┴"
        + "─" * (col_widths["status"] + 2)
        + "┴"
        + "─" * (col_widths["time"] + 2)
        + "┴"
        + "─" * (col_widths["details"] + 2)
        + "┘"
    )

    print("\n" + top_border)
    print(format_row("Example", "Status", "Time(s)", "Details"))
    print(separator)

    for result in results:
        status_colored = format_status(result.status)
        time_str = f"{result.execution_time:.2f}" if result.execution_time > 0 else "-"
        details = (
            result.details[: col_widths["details"] - 3] + "..."
            if len(result.details) > col_widths["details"]
            else result.details
        )
        print(format_row(result.name, status_colored, time_str, details))

    print(bottom_border)

    # Print summary
    summary_parts = []
    if success_count > 0:
        summary_parts.append(f"{Colors.GREEN}{success_count} SUCCESS{Colors.RESET}")
    if failed_count > 0:
        summary_parts.append(f"{Colors.RED}{failed_count} FAILED{Colors.RESET}")
    if skipped_count > 0:
        summary_parts.append(f"{Colors.YELLOW}{skipped_count} SKIPPED{Colors.RESET}")
    if timeout_count > 0:
        summary_parts.append(f"{Colors.BLUE}{timeout_count} TIMEOUT{Colors.RESET}")

    print(f"\nSummary: {', '.join(summary_parts)}")
    print(f"Total Time: {total_time:.2f}s")


def discover_examples(examples_dir: Path) -> list[str]:
    """Discover all example files in the examples directory.

    Args:
        examples_dir: Path to the examples directory

    Returns:
        Sorted list of example filenames (excluding __init__.py, utils.py, runner.py)
    """
    exclude_files = {"__init__.py", "utils.py", "runner.py"}

    example_files = []
    for file_path in examples_dir.glob("*.py"):
        if file_path.name not in exclude_files:
            example_files.append(file_path.name)

    return sorted(example_files)


def main():
    """Run all examples and display summary."""
    examples_dir = Path(__file__).parent

    print("CARL Examples Runner")
    print("=" * 60)

    example_files = discover_examples(examples_dir)
    print(f"Running {len(example_files)} examples...")
    print()

    results = []
    start_time = time.time()

    for example_file in example_files:
        print(f"▶ Running {example_file}...", end=" ", flush=True)
        result = run_example(example_file, examples_dir)
        results.append(result)

        # Format status with color for progress output
        if result.status == "SUCCESS":
            status_color = Colors.GREEN
        elif result.status == "FAILED":
            status_color = Colors.RED
        elif result.status == "SKIPPED":
            status_color = Colors.YELLOW
        elif result.status == "TIMEOUT":
            status_color = Colors.BLUE
        else:
            status_color = Colors.RESET

        print(f"{status_color}{result.status}{Colors.RESET} ({result.execution_time:.2f}s)")

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print_summary_table(results, total_time)

    # Return exit code based on results
    # Exit 0 if all SUCCESS or SKIPPED, 1 if any FAILED or TIMEOUT
    has_failures = any(r.status in ["FAILED", "TIMEOUT"] for r in results)
    sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    main()
