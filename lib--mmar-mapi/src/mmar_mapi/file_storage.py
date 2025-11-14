import json
import string
from datetime import datetime
from hashlib import md5
from pathlib import Path
from zipfile import ZipFile, is_zipfile

ResourceId = str
ASCII_DIGITS_SPECIAL = set(string.ascii_lowercase + string.digits + "-")
SUFFIX_DIR = ".dir"
SUFFIX_METADATA = ".metadata"


def _validate_exist(files_dir):
    if not files_dir.exists():
        err = f"Failed to access file-storage directory: {files_dir}"
        raise OSError(err)


def _validate_dtype(dtype: str):
    if all(map(ASCII_DIGITS_SPECIAL.__contains__, dtype)):
        return
    raise ValueError(f"Bad dtype: {dtype}")


def generate_fname(content, dtype):
    fname_hash = md5(content).hexdigest()
    fname = f"{fname_hash}.{dtype}"
    return fname


class FileStorage:
    def __init__(self, files_dir):
        self.files_dir = Path(files_dir)
        self.files_dir.mkdir(exist_ok=True, parents=True)
        _validate_exist(self.files_dir)

    def _generate_fname_path(self, content: bytes, dtype: str):
        fpath = self.files_dir / generate_fname(content, dtype)
        return fpath

    def upload_maybe(self, content: bytes | str | None, fname: str) -> ResourceId | None:
        if not content:
            return None
        resource_id = self.upload(content, fname)
        return resource_id

    def upload(self, content: bytes | str, fname: str, origin: str | None = None) -> ResourceId:
        if isinstance(content, str):
            content = content.encode()

        dtype = fname.rsplit(".", 1)[-1]
        _validate_dtype(dtype)
        fpath = self._generate_fname_path(content, dtype)
        fpath.write_bytes(content)

        fpath_md = fpath.with_suffix(SUFFIX_METADATA)
        update_date = f"{datetime.now():%Y-%m-%d--%H-%M-%S}"
        metadata = {"fname": fname, "update_date": update_date, "size": len(content), "origin": origin}
        fpath_md.write_text(json.dumps(metadata, ensure_ascii=False))

        return str(fpath)

    def get_metadata(self, resource_id: ResourceId) -> dict | None:
        metadata_path = Path(resource_id).with_suffix(SUFFIX_METADATA)
        if not metadata_path.exists():
            return None
        return json.loads(metadata_path.read_text())

    def get_fname(self, resource_id: ResourceId) -> str | None:
        metadata = self.get_metadata(resource_id)
        if metadata is None:
            return None
        return metadata.get("fname")

    async def upload_async(self, content: bytes | str, fname: str) -> ResourceId:
        return self.upload(content, fname)

    def upload_dir(self, resource_ids: list[ResourceId]) -> ResourceId:
        content = "\n".join(resource_ids)
        res = self.upload(content=content, fname=".dir")
        return res

    def download(self, resource_id: ResourceId) -> bytes:
        return Path(resource_id).read_bytes()

    async def download_async(self, resource_id: ResourceId) -> bytes:
        return self.download(resource_id)

    def download_text(self, resource_id: ResourceId) -> str:
        return Path(resource_id).read_text(encoding="utf-8")

    def read_dir_or_none(self, resource_id: ResourceId) -> list[ResourceId] | None:
        if not self.is_dir(resource_id):
            return None
        res = self.download_text(resource_id).split("\n")
        return res

    def get_path(self, resource_id: ResourceId | None) -> Path | None:
        return self._get_path(resource_id)

    def _get_path(self, resource_id: ResourceId | None) -> Path | None:
        if not resource_id:
            return None
        path = Path(resource_id)
        return path if (path.exists() and path.is_file()) else None

    def is_valid(self, resource_id: ResourceId | None) -> bool:
        path = self._get_path(resource_id)
        return path is not None

    def is_file(self, resource_id: ResourceId | None) -> bool:
        path = self._get_path(resource_id)
        return bool(path and path.suffix != SUFFIX_DIR)

    def is_dir(self, resource_id: ResourceId | None) -> bool:
        path = self._get_path(resource_id)
        return bool(path and path.suffix == SUFFIX_DIR)

    def get_dtype(self, resource_id: ResourceId | None) -> str | None:
        return resource_id and resource_id.rsplit(".")[-1].lower()

    def unzip_file(self, resource_id: str) -> ResourceId:
        """takes resource_id which refer to zip-archive, unpacks it and returns directory ResourceId with content of zip-archive"""
        path = self._get_path(resource_id)
        if not path:
            raise ValueError(f"Not found path: {resource_id}")
        if not is_zipfile(resource_id):
            raise ValueError(f"Expected zip archive but found: {resource_id}")

        resource_ids = []

        with ZipFile(path, mode="r") as zip_file:
            for file_info in zip_file.filelist:
                file_dtype = file_info.filename.rsplit(".")[-1]
                file_bytes = zip_file.read(file_info)
                rid = self.upload(file_bytes, file_dtype)
                resource_ids.append(rid)

        res = self.upload_dir(resource_ids)
        return res
