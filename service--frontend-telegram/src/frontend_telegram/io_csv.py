import csv
import os
from collections.abc import Callable
from typing import Awaitable

from loguru import logger

from frontend_telegram.io_fs import is_empty_file


def _get_header(csv_path: str) -> list[str]:
    """Return the header (list of column names) from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file '{csv_path}' is empty.")
        return header


def get_values_from_csv(csv_path: str, column_name: str) -> set[str]:
    """Return all values from the given column in the CSV file."""
    header = _get_header(csv_path)

    if column_name not in header:
        raise ValueError(f"Invalid column name '{column_name}'. Valid options: {header}")

    values = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            values.append(row[column_name])
    res = set(values)
    return res


def _validate_keys(row: dict, header: list[str]) -> None:
    if list(row.keys()) != header:
        raise ValueError(f"Row keys {list(row.keys())} do not match header {header}")


def validate_csv_header(csv_path: str, header: list[str]) -> None:
    csv_header = _get_header(csv_path)
    if csv_header != header:
        raise ValueError(f"Checking {csv_path}, expected header: {header}, found: {csv_header}")


def add_row_to_csv(csv_path: str, row: dict) -> None:
    """Add a row (dict) to the CSV file after validating header."""
    header = _get_header(csv_path)
    _validate_keys(row, header)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter=",")
        writer.writerow(row)


def touch_csv(csv_path: str, header: list[str]) -> None:
    """Ensure a CSV file exists with the given header.
    If it exists, validate that the header matches.
    """
    if os.path.exists(csv_path) and not is_empty_file(csv_path):
        existing_header = _get_header(csv_path)
        if existing_header != header:
            raise ValueError(f"Header mismatch in existing CSV.\nExpected: {header}\nFound:    {existing_header}")
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(header)


async def modify_csv(csv_path: str, mapper: Callable[[dict], Awaitable[dict | None]]) -> None:
    """Modify each row of the CSV using a mapper function.
    The mapper receives a row dict and must return a new dict with the same keys.

    Changes are written back to the same file.
    """
    header = _get_header(csv_path)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter=","))

    rows = [dict(row) for row in reader]
    modified_rows = [await mapper(dict(row)) for row in reader]
    if not any(modified_rows):
        logger.info(f"No changes in {csv_path}")
        return

    for mr in filter(None, modified_rows):
        _validate_keys(mr, header)

    final_rows = [new_row or row for row, new_row in zip(rows, modified_rows)]

    # Write back to the same file
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter=",")
        writer.writeheader()
        writer.writerows(final_rows)
