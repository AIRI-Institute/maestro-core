from pathlib import Path


def get_modified_time(fpath: Path) -> float:
    # can be moved to io_files
    if not fpath.exists():
        return -1.0
    return fpath.stat().st_mtime

def is_empty_file(fpath: Path | str) -> bool:
    fpath = Path(fpath)
    return fpath.stat().st_size < 2 and fpath.read_text().strip().__len__() == 0
