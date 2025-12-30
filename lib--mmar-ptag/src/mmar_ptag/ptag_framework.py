"""PTAG ~ 'Pydantic Type Adapter GRPC'"""

import types
from collections.abc import Callable
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Generic, TypeVar, cast

from google.protobuf.message import Message
from grpc import Channel, RpcError, ServicerContext, StatusCode, insecure_channel
from loguru import logger
from mmar_utils.utils_inspect import (
    FuncMetadata,
    Metadatas,
    bind_args_to_tuple,
    extract_and_validate_obj_methods_metadatas,
    extract_interface_metadatas,
    get_full_name,
    prettify_arg_metadata,
)

from .logging_configuration import TRACE_ID, TRACE_ID_DEFAULT
from .ptag_pb2 import PTAGRequest, PTAGResponse
from .ptag_pb2_grpc import PTAGServiceServicer, PTAGServiceStub, add_PTAGServiceServicer_to_server

T = TypeVar("T")
TRACE_ID_VAR: ContextVar[str] = ContextVar(TRACE_ID, default="")


@contextmanager
def installed_trace_id(trace_id):
    token = TRACE_ID_VAR.set(trace_id)
    try:
        yield
    finally:
        TRACE_ID_VAR.reset(token)

@contextmanager
def nothing(*args, **kwargs):
    yield
    

class TraceIdProxy(Generic[T]):
    def __init__(self, service: T):
        self.service = service
        
        _, metadatas = extract_and_validate_obj_methods_metadatas(service)
        check_valid_trace_id_in_metadatas(metadatas)
        self.metadatas = metadatas
        
        self._set_proxy_methods()
    
    def _set_proxy_methods(self):
        for mm in self.metadatas.values():
            proxy = self._make_trace_id_proxy(mm)
            bound_func = types.MethodType(proxy, self)
            setattr(self, mm.name, bound_func)
    
    def _make_trace_id_proxy(self, mm):
        def wrapped(proxy_self, *args, **kwargs):
            trace_id = kwargs.pop('trace_id', None)
            context = nothing() if trace_id is None else logger.contextualize(trace_id=trace_id)
            with context:
                return getattr(proxy_self.service, mm.name)(*args, **kwargs)
        return wrapped
    
    def __str__(self):
        return f"TraceIdProxy('{get_full_name(self.service)}')"


def check_valid_trace_id_in_func(fm: FuncMetadata):
    args_len = len(fm.args_metadata)
    for ii, am in enumerate(fm.args_metadata):
        if am.name != TRACE_ID:
            continue
        if ii != args_len - 1:
            raise ValueError(f"Expected {TRACE_ID} on last place, but found: {fm.as_pretty_str()}")
        if not am.default == "":
            raise ValueError(f'Expected `{TRACE_ID}: str=""`, found: {prettify_arg_metadata(am)}')


def check_valid_trace_id_in_metadatas(metadatas: Metadatas):
    for fm in metadatas.values():
        check_valid_trace_id_in_func(fm)


def as_str(obj: str | bytes) -> str:
    return obj if isinstance(obj, str) else obj.decode()


class WrappedPTAGService(PTAGServiceServicer):
    def __init__(self, service_object):
        self.methods, self.metadatas = extract_and_validate_obj_methods_metadatas(service_object)
        check_valid_trace_id_in_metadatas(self.metadatas)

    def Invoke(self, request: Message, context: ServicerContext):
        method_name = request.FunctionName
        method = self.methods.get(method_name)
        method_metadata = self.metadatas.get(method_name)

        if method_metadata is None:
            context.set_code(StatusCode.NOT_FOUND)
            context.set_details(f"Method {method_name} not found")
            return PTAGResponse()

        # [args_bytes] -(args_adapter.validate)-> [args] -(method)-> [result] -(result_adapter.dump)-> [result_bytes]
        args_adapter = method_metadata.args_adapter
        result_adapter = method_metadata.result_adapter

        metadata = dict(context.invocation_metadata())
        trace_id = as_str(metadata.get(TRACE_ID, TRACE_ID_DEFAULT))
        try:
            input_obj = args_adapter.validate_json(request.Payload)
            input_names = (am.name for am in method_metadata.args_metadata)
            input_kwargs = dict(zip(input_names, input_obj))

            with installed_trace_id(trace_id=trace_id):
                with logger.contextualize(trace_id=trace_id):
                    output_obj = method(**input_kwargs)

            payload = result_adapter.dump_json(output_obj)
            return PTAGResponse(FunctionName=method_name, Payload=payload)
        except Exception as e:
            with logger.contextualize(trace_id=trace_id):
                logger.exception(f"Failed to process request: {e}")
            context.set_code(StatusCode.INTERNAL)
            context.set_details(str(e))
            return PTAGResponse()


def make_proxy(grpc_stub, func_metadata: FuncMetadata):
    # only **kwargs supported
    # [args] -(args_adapter.dump)-> [args_bytes] -(send)-> [result_bytes] -(return_adapter.validate)-> [result]
    def proxy(self, *args, **kwargs):
        if args:
            raise ValueError(f"Func `{func_metadata.name}`: only kwargs supported, but args found: `{args}`")
        kw_get_or_pop = kwargs.get if func_metadata.has_arg(TRACE_ID) else kwargs.pop
        trace_id = kw_get_or_pop(TRACE_ID, None) or TRACE_ID_VAR.get()
        metadata = [(TRACE_ID, trace_id)] if trace_id else []

        args = bind_args_to_tuple(func_metadata.args_metadata, kwargs=kwargs)
        args_bytes = func_metadata.args_adapter.dump_json(args)
        request = PTAGRequest(FunctionName=func_metadata.name, Payload=args_bytes)
        response = grpc_stub.Invoke(request, metadata=metadata)
        result_bytes = response.Payload
        result = func_metadata.result_adapter.validate_json(result_bytes)
        return result

    return proxy


ChannelStubFunc = Callable[[str], tuple[Channel, PTAGServiceStub]]


def _create_insecure_channel_stub(address):
    channel = insecure_channel(address)
    stub = PTAGServiceStub(channel)
    return channel, stub


class ClientProxy(Generic[T]):
    def __init__(self, service_interface: type[T], address, channel_stub_func: ChannelStubFunc | None = None):
        self.channel_stub_func = channel_stub_func or _create_insecure_channel_stub
        if not isinstance(service_interface, type):
            si_name = type(service_interface).__name__
            raise ValueError(
                f"Expected type, found: {type(service_interface)}. Probably you passed ptag_client({si_name}(), ...) instead of ptag_client({si_name}, ...)"
            )
        self.service_interface = service_interface

        metadatas = extract_interface_metadatas(service_interface)
        check_valid_trace_id_in_metadatas(metadatas)
        self.metadatas = metadatas

        self.address = address
        self._channel, self._stub = self.channel_stub_func(self.address)
        self._set_proxy_methods()

    def _set_proxy_methods(self):
        for mm in self.metadatas.values():
            proxy = make_proxy(self._stub, mm)
            proxy_wrapped = self._wrap_method_with_reconnect(proxy)
            bound_func = types.MethodType(proxy_wrapped, self)
            setattr(self, mm.name, bound_func)

    # fix, there is 2 responsibilities now: reconnect and return method
    def _reconnect(self, method_name):
        try:
            self._channel.close()
        except Exception:
            pass
        self._channel, self._stub = self.channel_stub_func(self.address)
        self._set_proxy_methods()
        logger.info(f"Reconnected on {self.address}...")
        return getattr(self, method_name)

    def _wrap_method_with_reconnect(self, raw_method):
        def wrapped(proxy_self, *args, **kwargs):
            try:
                return raw_method(proxy_self, *args, **kwargs)
            except RpcError as ex:
                if ex.code() == StatusCode.UNAVAILABLE:
                    logger.error(f"ERROR: [{ex.details()}], reconnecting...")
                    fresh_method = proxy_self._reconnect(raw_method.__name__)
                    return fresh_method(*args, **kwargs)
                raise
            except ValueError as ex:
                if str(ex) == 'Cannot invoke RPC on closed channel!':
                    logger.error(f"ERROR: [{ex}], reconnecting...")
                    fresh_method = proxy_self._reconnect(raw_method.__name__)
                    return fresh_method(*args, **kwargs)
                raise

        return wrapped

    def __str__(self):
        return f"ptag-client('{get_full_name(self.service_interface)}' -> '{self.address}')"


def ptag_attach(server, service_object):
    """
    Attach a service object implementing the interface to a gRPC server.
    """
    service = WrappedPTAGService(service_object)
    add_PTAGServiceServicer_to_server(service, server)


def _is_valid_address(address) -> bool:
    if ":" not in address:
        return False
    host, port = address.rsplit(":", 1)
    if not port.isnumeric():
        return False
    return True


def _try_fix_address(address: int | str):
    if isinstance(address, int) or address.isnumeric():
        return f"0.0.0.0:{address}"
    if address.startswith(":") and address[1:].isnumeric():
        return f"0.0.0.0{address}"
    if not _is_valid_address(address):
        logger.warning(f"Probably invalid address passed: {address}")
    return address


def ptag_client(service_interface: type[T], address: str) -> T:
    """
    Create a dynamic client for the given interface at the provided gRPC address.
    """
    address = _try_fix_address(address)
    proxy = ClientProxy(service_interface, address)
    return cast(T, proxy)
