# mmar-ptag

pydantically-type-adapted-grpc framework for multimodal architectures team

**The main goal:** simplify defining type-safe Python interfaces to separate services.

## Installation

```bash
pip install mmar-ptag
```

## Quick Start

### 1. Define the Service

```python
from types import SimpleNamespace
from mmar_ptag import deploy_server

class Greeter:
    def say_hello(self, *, name: str, count: int = 1, trace_id: str = "") -> dict:
        return {"message": f"Hello, {name} x{count}"}
```

### 2. Run the Server

```python
deploy_server(
    config_server=SimpleNamespace(port=50051, max_workers=10),
    service=Greeter()
)
```

### 3. Create a Client

```python
from mmar_ptag import ptag_client

class Greeter:
    def say_hello(self, *, name: str, count: int = 1) -> dict:
        ...

client = ptag_client(Greeter, "localhost:50051")
result = client.say_hello(name="World", count=3)
print(result)  # {'message': 'Hello, World x3'}
```

## Trace ID Support

The `trace_id` parameter enables distributed tracing across your service mesh. When included in a method signature, it is automatically propagated through the call chain.

```python
from mmar_ptag import ptag_client

class UserService:
    def get_user(self, *, user_id: int, trace_id: str = "") -> dict:
        ...

client = ptag_client(UserService, "localhost:50051")

# Trace ID is automatically propagated through the call chain
result = client.get_user(user_id=123, trace_id="request-abc-123")
```

The `trace_id` value is stored in `mmar_mimpl.TRACE_ID_VAR`, making it accessible throughout the request lifecycle without manual passing.

## How It Works

ptag uses Pydantic adapters to handle type conversion between Python and gRPC/protobuf.

**Example: `client.say_hello(name="World", count=3)`**

**Client flow (sending request):**
```
{name="World", count=3}
    -(args tuple)->
("World", 3)
    -(args_adapter.dump_json)->
["World", 3]
    -(wrap in PTAGRequest)->
b'\n\tsay_hello\x12\x0c["World", 3]'
    -(server receives)->
```

**Server flow (processing & responding):**
```
b'\n\tsay_hello\x12\x0c["World", 3]'
    -(args_adapter.validate_json)->
["World", 3]
    -(bind to kwargs)->
{name="World", count=3}
    -(say_hello method)->
{"message": "Hello, World x3"}
    -(result_adapter.dump_json)->
{"message": "Hello, World x3"}
    -(wrap in PTAGResponse)->
b'\n\tsay_hello\x12\x1e{"message": "Hello, World x3"}'
    -(client receives)->
```

**Client flow (receiving response):**
```
b'\n\tsay_hello\x12\x1e{"message": "Hello, World x3"}'
    -(return_adapter.validate_json)->
{"message": "Hello, World x3"}
```

This ensures:
- Arguments are validated and serialized before sending
- Return values are deserialized and validated after receiving
- Type safety across the wire without manual protobuf definitions

## Features

- **Type-safe RPC** using Pydantic for validation
- **Automatic reconnection** with configurable retry attempts
- **Built-in tracing** with trace ID support
- **Interface-based design** – define services as Python classes
- **Keyword-only arguments** – explicit and readable API

## API Reference

### `ptag_client(interface, address, reconnect_attempts=5)`

Create a dynamic client for the given interface at the provided gRPC address.

- `interface`: Type (class) defining the service interface
- `address`: gRPC server address (e.g., `"localhost:50051"`)
- `reconnect_attempts`: Number of retry attempts on connection failure (default: 5)

### `ptag_attach(server, service_object)`

Attach a service object to an existing gRPC server.
