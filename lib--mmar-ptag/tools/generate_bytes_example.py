"""Generate actual protobuf bytes for the README example.

This script demonstrates the exact byte serialization that happens in ptag
for the Greeter.say_hello(name="World", count=3) example.
"""

import json
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mmar_ptag.ptag_pb2 import PTAGRequest, PTAGResponse


def main():
    print("=" * 60)
    print("PTAG Byte Serialization Example")
    print("=" * 60)

    # Example: say_hello(name="World", count=3)
    name = "World"
    count = 3

    print("\n--- CLIENT SIDE: Sending Request ---")
    print(f"Python kwargs: {{name={repr(name)}, count={count}}}")

    # Args are converted to tuple by bind_args_to_tuple
    args_tuple = (name, count)
    print(f"Args tuple: {args_tuple}")

    # Args are serialized to JSON by args_adapter.dump_json
    args_json = json.dumps(args_tuple)
    args_bytes = args_json.encode('utf-8')
    print(f"Args JSON: {repr(args_json)}")
    print(f"Args bytes: {repr(args_bytes)}")

    # Wrap in PTAGRequest protobuf
    request = PTAGRequest(FunctionName="say_hello", Payload=args_bytes)
    request_bytes = request.SerializeToString()
    print(f"\nPTAGRequest bytes: {repr(request_bytes)}")
    print(f"PTAGRequest hex: {request_bytes.hex()}")

    # Decode the protobuf for explanation
    print("\nProtobuf field breakdown:")
    print(f"  Field 1 (FunctionName): tag=0x0a (field 1, string type)")
    print(f"    Length: {len('say_hello')} = 0x{len('say_hello'):x}")
    print(f"    Value: 'say_hello'")
    print(f"  Field 2 (Payload): tag=0x12 (field 2, bytes type)")
    print(f"    Length: {len(args_bytes)} = 0x{len(args_bytes):x}")
    print(f"    Value: {repr(args_bytes)}")

    print("\n--- SERVER SIDE: Processing Request ---")
    # Server deserializes
    parsed_request = PTAGRequest()
    parsed_request.ParseFromString(request_bytes)
    print(f"Parsed FunctionName: {parsed_request.FunctionName}")
    print(f"Parsed Payload: {repr(parsed_request.Payload)}")

    # Server validates JSON back to tuple
    parsed_args = json.loads(parsed_request.Payload)
    print(f"Args after JSON parse: {parsed_args}")

    # Server executes method
    result_dict = {"message": f"Hello, {name} x{count}"}
    print(f"Method result: {result_dict}")

    # Result serialized to JSON by result_adapter.dump_json
    result_json = json.dumps(result_dict)
    result_bytes = result_json.encode('utf-8')
    print(f"Result JSON: {repr(result_json)}")
    print(f"Result bytes: {repr(result_bytes)}")

    # Wrap in PTAGResponse protobuf
    response = PTAGResponse(FunctionName="say_hello", Payload=result_bytes)
    response_bytes = response.SerializeToString()
    print(f"\nPTAGResponse bytes: {repr(response_bytes)}")
    print(f"PTAGResponse hex: {response_bytes.hex()}")

    print("\nProtobuf field breakdown:")
    print(f"  Field 1 (FunctionName): tag=0x0a (field 1, string type)")
    print(f"    Length: {len('say_hello')} = 0x{len('say_hello'):x}")
    print(f"    Value: 'say_hello'")
    print(f"  Field 2 (Payload): tag=0x12 (field 2, bytes type)")
    print(f"    Length: {len(result_bytes)} = 0x{len(result_bytes):x}")
    print(f"    Value: {repr(result_bytes)}")

    print("\n--- CLIENT SIDE: Receiving Response ---")
    parsed_response = PTAGResponse()
    parsed_response.ParseFromString(response_bytes)
    print(f"Parsed Payload: {repr(parsed_response.Payload)}")

    # Client validates JSON back to dict
    final_result = json.loads(parsed_response.Payload)
    print(f"Final result: {final_result}")

    print("\n" + "=" * 60)
    print("SUMMARY for README")
    print("=" * 60)
    print(f"\nRequest bytes: {repr(request_bytes)}")
    print(f"Response bytes: {repr(response_bytes)}")


if __name__ == "__main__":
    main()
