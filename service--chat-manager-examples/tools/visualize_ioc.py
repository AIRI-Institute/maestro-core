#!/usr/bin/env python
"""
Script to visualize the IOC dependency graph using dishka's built-in renderers.

Usage:
    python scripts/visualize_ioc.py --format mermaid
    python scripts/visualize_ioc.py --format d2
"""

import argparse
from pathlib import Path

from dishka import make_container
from dishka.plotter import render_d2, render_mermaid

from chat_manager_examples.ioc import IOCS


def main():
    parser = argparse.ArgumentParser(description="Visualize IOC dependency graph")
    parser.add_argument(
        "--format",
        choices=["mermaid", "d2"],
        default="mermaid",
        help="Output format (default: mermaid)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (prints to stdout if not specified)",
    )
    args = parser.parse_args()

    # Create container
    container = make_container(*[ioc() for ioc in IOCS])

    # Render graph
    if args.format == "mermaid":
        output = render_mermaid(container)
    else:  # d2
        output = render_d2(container)

    # Output
    if args.output:
        args.output.write_text(output)
        print(f"Graph written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
