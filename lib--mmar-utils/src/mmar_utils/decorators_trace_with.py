import time
import traceback
import warnings
from datetime import UTC
from datetime import datetime as dt
from typing import Any, Callable, Literal

from pydantic import BaseModel, field_serializer

from .mmar_types import Either, Result
from .utils_inspect import bind_args_to_dict, bind_args_to_tuple, extract_func_metadata

FunInput = dict | tuple | bytes | str
FunOutput = str | bytes | dict | BaseModel
InputType = Literal[dict, tuple, bytes] | Callable
OutputType = Literal[Any, str, bytes] | Callable


class ExceptionInfo(BaseModel):
    clazz: str
    message: str
    stacktrace: str


class FunctionEnter(BaseModel):
    namespace: str
    call_datetime: dt
    fun_input: FunInput

    @field_serializer("call_datetime")
    def serialize_dt(self, value: dt, _info):
        return value.isoformat()


class FunctionInvocation(BaseModel):
    elapsed_seconds: float | None = None
    fun_result: Either[ExceptionInfo, FunOutput]


class FunctionCall(BaseModel):
    enter: FunctionEnter
    invocation: FunctionInvocation | None


def transform_fun_input(fn_metadata, *, args, kwargs, input_as: InputType, validate_ext: bool):
    bind_kwargs = dict(args_metadata=fn_metadata.args_metadata, args=args, kwargs=kwargs, validate_ext=validate_ext)

    if input_as is bytes:
        fun_args = bind_args_to_tuple(**bind_kwargs)
        res = fn_metadata.args_adapter.dump_json(fun_args)
    elif input_as is dict:
        res = bind_args_to_dict(**bind_kwargs)
    elif input_as is tuple:
        res = bind_args_to_tuple(**bind_kwargs)
    elif isinstance(input_as, callable):
        res = input_as(args, kwargs)
    else:
        raise ValueError(f"Unsupported input_as: {input_as}")
    return res


def transform_fun_output(fn_metadata, result, output_as: OutputType):
    if output_as is Any:
        res = result
    elif output_as is bytes:
        res = fn_metadata.result_adapter.dump_json(result)
    elif output_as is str:
        res = str(result)
    elif callable(output_as):
        res = output_as(result)
    else:
        raise ValueError(f"Unsupported output_as: {output_as}")
    return res


def trace_with(
    spy: Callable[[FunctionCall], None],
    namespace: str | None = None,
    trace_enters: bool = False,
    input_as: InputType = dict,
    output_as: OutputType = Any,
    validate_ext: bool = False,
):
    def decorator(fn: Callable[[...], Result]):
        ns = namespace or fn.__qualname__

        def run_spy(fun_enter, fun_invocation: FunctionInvocation | None):
            fc = FunctionCall(enter=fun_enter, invocation=fun_invocation)
            try:
                spy(fc)
            except Exception as ex:
                warnings.warn(f"Spy function failed: {ex}", stacklevel=2)

        fn_metadata = extract_func_metadata(fn, only_kw=False)
        if fn_metadata is None:
            raise ValueError(f"Failed to parse function: {fn}")

        def wrapper(*args, **kwargs):
            call_dt = dt.now(UTC)

            fun_input = transform_fun_input(
                fn_metadata, args=args, kwargs=kwargs, input_as=input_as, validate_ext=validate_ext
            )
            fun_enter = FunctionEnter(namespace=ns, call_datetime=call_dt, fun_input=fun_input)
            if trace_enters:
                run_spy(fun_enter, None)

            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed_seconds = time.perf_counter() - start
                fun_output = transform_fun_output(fn_metadata, result=result, output_as=output_as)
                fun_invocation = FunctionInvocation(elapsed_seconds=elapsed_seconds, fun_result=(None, fun_output))
                run_spy(fun_enter, fun_invocation)
                return result
            except Exception as ex:
                elapsed_seconds = time.perf_counter() - start
                exc_info = ExceptionInfo(
                    clazz=ex.__class__.__name__,
                    message=str(ex),
                    stacktrace="".join(traceback.format_exception(type(ex), ex, ex.__traceback__)),
                )
                fun_invocation = FunctionInvocation(elapsed_seconds=elapsed_seconds, fun_result=(exc_info, None))
                run_spy(fun_enter, fun_invocation)
                raise

        return wrapper

    return decorator
