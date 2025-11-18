from mmar_mapi import Context

from src.fastapi_errors import MalformedException, WrongClientException


def validate_client(client_id: str) -> str:
    if "_" in client_id:
        raise WrongClientException()
    return client_id


def validate_no_underscores(context: Context, field: str) -> None:
    value = getattr(context, field, "")
    if "_" in value:
        raise MalformedException(f"Not supported underscores (`_`) in {field}")
