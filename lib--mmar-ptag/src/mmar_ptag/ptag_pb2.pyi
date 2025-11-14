from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PTAGRequest(_message.Message):
    __slots__ = ("FunctionName", "Payload")
    FUNCTIONNAME_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    FunctionName: str
    Payload: bytes
    def __init__(self, FunctionName: _Optional[str] = ..., Payload: _Optional[bytes] = ...) -> None: ...

class PTAGResponse(_message.Message):
    __slots__ = ("FunctionName", "Payload")
    FUNCTIONNAME_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    FunctionName: str
    Payload: bytes
    def __init__(self, FunctionName: _Optional[str] = ..., Payload: _Optional[bytes] = ...) -> None: ...
