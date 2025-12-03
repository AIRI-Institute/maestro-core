import functools
import time
from typing import Protocol


class LoggerI(Protocol):
    def info(msg: str):
        pass

    def exception(msg: str):
        pass


# todo eliminate: move to mmar_utils
def trace_duration(logger: LoggerI, label: str | None = None, show_args: bool = False):
    """
    Generic timing/logging decorator with 'Request ...' format.

    Args:
        label: Optional label instead of function name.
        show_args: If True, include function args/kwargs in parentheses.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fn_label = label or func.__name__
            start = time.time()
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.exception(f"Request {fn_label}: failed to process")
                return None
            finally:
                elapsed = time.time() - start
                args_str = ""
                if show_args:
                    # Simple representation of args
                    args_list = [repr(a) for a in args]
                    kwargs_list = [f"{k}={v!r}" for k, v in kwargs.items()]
                    all_args = args_list + kwargs_list
                    args_str = "(" + ", ".join(all_args) + ")"
                msg = f"Request {fn_label}{args_str}: processed in {elapsed:.2f} seconds"
                if elapsed > 60:
                    msg += f" (~{elapsed / 60:.2f} minutes)"
                logger.info(msg)

        return wrapper

    return decorator


from mmar_mapi.services import DocExtractionOutput
from more_itertools import flatten


# todo eliminate: move to mmar-mapi
def merge_outputs(outputs: list[DocExtractionOutput]) -> DocExtractionOutput:
    assert outputs
    # todo fix mmar-mapi: make engine ExtractionEngineSpec frozen, then eliminate `model_dump_json`
    assert set(out.spec.engine.model_dump_json() for out in outputs).__len__() == 1
    assert all(out.spec.page_range for out in outputs)
    # todo validate outputs, page_ranges should be aligned
    # todo validate order
    if len(outputs) == 1:
        return outputs[0]
    out_fst = outputs[0]
    out_lst = outputs[-1]

    page_min = out_fst.spec.page_range[0]
    page_max = out_lst.spec.page_range[1]
    # todo fix: validate that page_range covers all pages
    page_range = (page_min, page_max)

    text = "\n".join(out.text for out in outputs if out.text)
    tables = list(flatten(out.tables for out in outputs))
    pictures = list(flatten(out.pictures for out in outputs))
    page_images = list(flatten(out.page_images for out in outputs))

    res = DocExtractionOutput(
        spec=out_fst.spec.with_page_range(page_range),
        text=text,
        tables=tables,
        pictures=pictures,
        page_images=page_images,
    )
    return res

from more_itertools import divide


def _validate_page_range(v: tuple[int, int]) -> tuple[int, int]:
    if v[0] < 1 or v[1] < v[0]:
        raise ValueError("Invalid page range: start must be ≥ 1 and end must be ≥ start.")
    return v


PageRange = tuple[int, int]  # start from 1, both inclusive


# todo eliminate: move to mmar_mapi
def split_range(rng: PageRange, chunks: int) -> list[PageRange]:
    _validate_page_range(rng)

    items = range(rng[0], rng[1] + 1)
    res = []
    for part_it in divide(chunks, items):
        part = list(part_it)
        if not part:
            break
        res.append((part[0], part[-1]))
    return res
