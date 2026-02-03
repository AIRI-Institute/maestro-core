from mimetypes import guess_type

from aiohttp import ClientError, ClientSession, ClientTimeout, FormData

from mmar_mcli.models import FileData


def make_file_form_data(file_data: FileData) -> FormData:
    fd = FormData()
    fname, fbytes = file_data
    content_type = guess_type(fname)[0] or "application/octet-stream"
    fd.add_field(name="file", filename=fname, content_type=content_type, value=fbytes)
    return fd


async def request_with_session(
    *,
    method: str,
    url: str,
    json: dict | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    timeout: int | None = None,
    data: FormData | None = None,
    headers_extra: dict[str, str] | None = None,
) -> bytes | dict:
    headers_all = headers | (headers_extra or {})
    async with ClientSession(headers={}) as session:
        timeout_ = ClientTimeout(timeout) if isinstance(timeout, int) else None
        async with session.request(
            method=method,
            url=url,
            json=json,
            headers=headers_all,
            params=params,
            timeout=timeout_,
            data=data,
        ) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            if "application/json" in content_type:
                body: dict = await resp.json()
                try:
                    resp.raise_for_status()
                except Exception as ex:
                    raise ClientError(f"{ex}\nResponse body: {body}") from ex
                return body
            else:
                return await resp.read()
