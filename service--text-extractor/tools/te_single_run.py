import json
import os
import time
from pathlib import Path

import fire
from dotenv import load_dotenv
from mmar_mapi import FileStorage
from mmar_mapi.services import TextExtractorAPI
from mmar_ptag import ptag_client

from text_extractor.config import Config
from text_extractor.text_extractor import TextExtractor

load_dotenv()


def process_document(
    address: str,
    doc_path: str,
    files_dir: str,
) -> dict:
    if address is not None:
        te = ptag_client(TextExtractorAPI, address)
        files_dir = os.getenv("FILES_DIR_DEFAULT", "/mnt/data/maestro/files")
    else:
        config = Config(files_dir=files_dir)
        te = TextExtractor(config)

    file_storage = FileStorage(files_dir)
    doc = Path(doc_path)
    image_bytes = doc.read_bytes()
    dtype = doc.suffix[1:]
    fpath = file_storage._generate_fname_path(image_bytes, dtype)
    if Path(fpath).exists():
        resource_id = str(fpath)
    else:
        resource_id = file_storage.upload(image_bytes, dtype)

    local_files_dir_prefix = os.getenv("LOCAL_FILES_DIR_PREFIX", "/mnt/data")
    resource_id = resource_id.replace(local_files_dir_prefix, "/mnt/data")

    start_time = time.time()
    result_resource_id = te.extract(resource_id=resource_id)
    elapsed = time.time() - start_time

    local_result_path = result_resource_id.replace("/mnt/data", local_files_dir_prefix)
    with open(local_result_path, "r") as f:
        extracted_text = f.read()

    return {
        "document": doc_path,
        "resource_id": resource_id,
        "result_resource_id": result_resource_id,
        "elapsed": elapsed,
        "extracted_text": extracted_text,
    }


def _save_results_json(result: dict, path: str) -> None:
    output_path = Path(path)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")


def main(
    doc_path: str = "resources/test.pdf",
    output_path: str = "test.json",
    files_dir: str = "/tmp",
    remote: str | bool = "",
):
    if not Path(doc_path).is_file():
        raise ValueError(f"doc-path: expected existing file, found: {doc_path}")
    if not output_path.endswith(".json"):
        raise ValueError(f"Expected json output path passed")

    # reusing remote as bool
    if isinstance(remote, bool) and remote is True:
        address = 'localhost:9691'
    elif isinstance(remote, str) and remote:
        address = remote
    else:
        address = None

    print(f'doc_path: {doc_path}')
    print(f'output_path: {output_path}')
    print(f'service: {address or "local"}')

    result = process_document(address=address, doc_path=doc_path, files_dir=files_dir)

    result_json = json.dumps(result, indent=2, ensure_ascii=False)
    Path(output_path).write_text(result_json)
    result_sz = len(result_json)
    print(f"Done! Processed '{doc_path}' and saved result (sz={result_sz}) to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
