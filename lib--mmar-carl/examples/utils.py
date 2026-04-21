"""Utility functions for CARL examples."""

# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"


def format_status(success: bool) -> str:
    """Format execution status with color.

    Args:
        success: Whether the execution was successful

    Returns:
        Colored status string (green SUCCESS or red FAILED)
    """
    status = "SUCCESS" if success else "FAILED"
    color = Colors.GREEN if success else Colors.RED
    return f"{color}{status}{Colors.RESET}"
