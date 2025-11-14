from pathlib import Path


def ensure_existing_dir(path: Path | str) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if path.is_file():
        raise ValueError(f"Path {path} is file")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path
