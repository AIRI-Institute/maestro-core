import inspect
import warnings
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Callable, get_type_hints, TypeVar, get_args, get_origin, Any

from pydantic import TypeAdapter, RootModel, ValidationError


T = TypeVar("T")
ArgName = str
empty = inspect.Parameter.empty
TYPES_CACHE = {}


@dataclass
class ArgMetadata:
    name: str
    typehint: type
    default: Any
    is_kw: bool


def prettify_type(t: Any) -> str:
    origin = get_origin(t)
    if origin is None:
        return t.__name__ if hasattr(t, "__name__") else str(t)
    args = get_args(t)
    if args:
        return f"{origin.__name__}[{', '.join(prettify_type(a) for a in args)}]"
    else:
        return origin.__name__


def prettify_arg_metadata(arg_metadata: ArgMetadata) -> str:
    name, arg_type, default = arg_metadata.name, arg_metadata.typehint, arg_metadata.default
    arg_type = getattr(arg_type, "__name__", arg_type)
    if default is empty:
        return f"{name}: {arg_type}"
    if default == "":
        default = "''"
    return f"{name}: {arg_type}={default.__repr__()}"


def prettify_args_metadata(args_metadata: list[ArgMetadata]) -> str:
    return ", ".join(prettify_arg_metadata(am) for am in args_metadata).strip()


@dataclass
class FuncMetadata:
    name: str
    has_self: bool
    args_metadata: list[ArgMetadata]
    result_type: type

    @cached_property
    def args_type(self) -> type[tuple]:
        return tuple[*(am.typehint for am in self.args_metadata)]

    @cached_property
    def args_adapter(self) -> TypeAdapter:
        return TypeAdapter(self.args_type)

    @cached_property
    def result_adapter(self) -> TypeAdapter:
        return TypeAdapter(self.result_type)

    def has_arg(self, arg_name) -> bool:
        return any(am.name == arg_name for am in self.args_metadata)

    def as_pretty_str(self):
        pretty_args = prettify_args_metadata(self.args_metadata)
        # todo support non-kw args in the future
        assert all(arg.is_kw for arg in self.args_metadata)
        pretty_res = prettify_type(self.result_type)
        self_maybe = "self, " if self.has_self else ""
        if pretty_args:
            return f"def {self.name}({self_maybe}*, {pretty_args}) -> {pretty_res}: ..."
        else:
            return f"def {self.name}({self_maybe}*) -> {pretty_res}: ..."

Methods = dict[str, Callable]
Metadatas = dict[str, FuncMetadata]


def _parse_param(param: inspect.Parameter) -> ArgMetadata | None:
    name = param.name
    param_type = param.annotation
    if param_type == empty:
        raise ValueError(f"Not found type for parameter `{name}`")
    # todo validate param_type: allow only builtins and pydantic
    default = param.default
    if default != empty and not isinstance(default, param_type):
        raise ValueError(f"For argument `{name}` type {param_type} is not aligned with default value: {default}")
    arg_is_kw = param.kind == inspect.Parameter.KEYWORD_ONLY
    return ArgMetadata(name=name, typehint=param_type, default=default, is_kw=arg_is_kw)


def is_class_function(func):
    if not inspect.isfunction(func):
        return False
    return "." in func.__qualname__


def _parse_args(func: Callable) -> list[ArgMetadata]:
    signature = inspect.signature(func)
    parameters_all = list(signature.parameters.values())
    if is_class_function(func):
        if parameters_all[0].name != "self":
            raise ValueError(f"Method with first `self` parameter expected, found: {parameters_all}")
        parameters = parameters_all[1:]
    else:
        parameters = parameters_all
    args_metadata_0 = [_parse_param(param) for param in parameters]
    args_metadata = [am for am in args_metadata_0 if am]

    return args_metadata


@lru_cache(None)
def get_pydantic_type(typehint: Any) -> type[RootModel]:
    if typehint not in TYPES_CACHE:
        TYPES_CACHE[typehint] = type("TmpModel", (RootModel,), {"__root_type__": typehint})
    return TYPES_CACHE[typehint]


def isinstance_ext(value: Any, typehint: Any, validate_ext: bool = False) -> bool:
    try:
        return isinstance(value, typehint)
    except TypeError:
        pass
    if not validate_ext:
        return True
    pydantic_model = get_pydantic_type(typehint)
    try:
        pydantic_model.model_validate(value)
        return True
    except ValidationError:
        return False


def bind_args_to_tuple(
    args_metadata: list[ArgMetadata],
    *,
    args: tuple | None = None,
    kwargs: dict | None = None,
    validate_ext: bool = False,
) -> tuple:
    args = args or ()
    kwargs = kwargs or {}

    excess_args = set(kwargs) - set(am.name for am in args_metadata)
    if excess_args:
        raise ValueError(f"Unexpected excess arguments passed: {excess_args}")

    bound_args = []
    for ii, am in enumerate(args_metadata):
        name, arg_type = am.name, am.typehint
        # todo respect only_kw here
        if ii < len(args):
            if am.name in kwargs:
                # when calling, this is TypeError, but during binding ValueError looks more suitable
                raise ValueError(f"Multiple values passed for argument {am.name}")
            arg_val = args[ii]
        elif am.name in kwargs:
            arg_val = kwargs[am.name]
        else:
            if am.default is empty:
                raise ValueError(f"Argument `{name}`: not found")
            arg_val = am.default

        if not isinstance_ext(arg_val, arg_type, validate_ext):
            warnings.warn(f"Bad arg_val: {arg_val}")
            raise ValueError(f"Argument for `{name}` (of type `{type(arg_val)}`) not aligned with type `{arg_type}`")

        bound_args.append(arg_val)
    res = tuple(bound_args)
    return res


def bind_args_to_dict(
    args_metadata: list[ArgMetadata],
    *,
    args: tuple | None = None,
    kwargs: dict | None = None,
    validate_ext: bool = False,
) -> dict:
    bound_args = bind_args_to_tuple(args_metadata, args=args, kwargs=kwargs, validate_ext=validate_ext)
    assert len(bound_args) == len(args_metadata)
    res = {am.name: ba for am, ba in zip(args_metadata, bound_args)}
    return res


def validate_only_kw(args_metadata):
    for am in args_metadata:
        if am.is_kw:
            continue
        am_pretty = prettify_args_metadata(args_metadata)
        signature = f"(self, *, {am_pretty})" if am_pretty else '(self)'
        param = prettify_arg_metadata(am)
        msg_parts = [
            f"Keyword-Only parameters expected, found positional: `{param}`.",
            f"Probably enough to fix signature to `{signature}`.",
            "See https://peps.python.org/pep-3102/ for additional info",
        ]
        msg = " ".join(msg_parts)
        raise ValueError(msg)


def extract_func_metadata(func: Callable, only_kw: bool = True) -> FuncMetadata | None:
    """
    Extract a func's type hints and return Pydantic adapters for the argument and return value.
    """
    func_name = func.__name__
    if func_name.startswith("_"):
        return None

    type_hints = get_type_hints(func)
    args_metadata = _parse_args(func)
    if only_kw:
        validate_only_kw(args_metadata)

    result_type = type_hints.get("return")
    if not result_type:
        raise ValueError(f"return type annotation for func `{func_name}` should present but not found!")
    func_metadata = FuncMetadata(
        name=func_name,
        has_self=is_class_function(func),
        args_metadata=args_metadata,
        result_type=result_type,
    )
    return func_metadata


def extract_method_metadata(method: Callable) -> FuncMetadata | None:
    func = getattr(method, '__func__', None)
    if not func:
        return None
    return extract_func_metadata(func)


def extract_interface_metadatas(interface) -> Metadatas:
    metadatas = {}
    for name, func in inspect.getmembers(interface, predicate=inspect.isfunction):
        func_metadata = extract_func_metadata(func)
        if name.startswith('_'):
            continue
        if func_metadata is None:
            raise ValueError(f"Failed to parse interface func: {func.__name__}")
        metadatas[name] = func_metadata
    return metadatas


def _extract_obj_methods_metadatas(obj, skip_inherited: bool=True) -> tuple[Methods, Metadatas]:
    methods, metadatas = {}, {}
    for name, method in inspect.getmembers(obj, predicate=inspect.ismethod):
        if skip_inherited and name not in obj.__class__.__dict__:
            continue

        # also can see only on func from the base interface class
        func = method.__func__
        func_metadata = extract_func_metadata(func)
        if func_metadata:
            methods[name] = method
            metadatas[name] = func_metadata
    return methods, metadatas


def _get_interface(obj):
    bases = obj.__class__.__bases__
    if len(bases) != 1:
        raise ValueError(f"Expected one base class, found: {bases}")
    interface = bases[0]
    return interface


def _get_full_class_name(cls):
    return cls.__module__ + "." + cls.__qualname__


def extract_and_validate_obj_methods_metadatas(obj):
    interface = _get_interface(obj)
    metadatas_i = extract_interface_metadatas(interface)

    methods, metadatas = _extract_obj_methods_metadatas(obj)

    if metadatas != metadatas_i:
        s_pref = f"service `{_get_full_class_name(type(obj))}`"
        i_pref = f"interface `{_get_full_class_name(interface)}`"
        sz = max(len(s_pref), len(i_pref))
        s_pref = s_pref.ljust(sz)
        i_pref = i_pref.ljust(sz)
        resolutions = {fn: (' # OK' if metadatas.get(fn) == metadatas_i.get(fn) else '') for fn in metadatas.keys()}
        metadatas_pretty = [f'{md.as_pretty_str()}{resolutions[fn]}' for fn, md in metadatas.items()]
        metadatas_i_pretty = [mdi.as_pretty_str() for mdi in metadatas_i.values()]

        msg_lines = [
            "Signatures mismatch between:",
            f"{s_pref} ::",
            *["\t" + md for md in metadatas_pretty],
            "and",
            f"{i_pref} ::",
            *["\t" + mdi for mdi in metadatas_i_pretty],
        ]
        msg = "\n".join(msg_lines)
        raise ValueError(msg)
    return methods, metadatas


def get_full_name(obj) -> str:
    # module
    if hasattr(obj, "__spec__") or hasattr(obj, "__package__"):
        return getattr(obj, "__name__", str(obj))

    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)

    if module and qualname:
        return f"{module}.{qualname}"
    elif qualname:
        return qualname
    elif module:
        return module
    else:
        return str(obj)
