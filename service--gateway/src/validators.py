from src.fastapi_errors import WrongClientException


def validate_client(client_id: str) -> str:
    if "_" in client_id:
        raise WrongClientException()
    return client_id
