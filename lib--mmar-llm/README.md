# mmar-llm

## how to run tests via pytest
1. Create `.env` in current directory.

Example:
```env
llm_config_path=/mnt/data/envs/creds/llm_config.json
test_endpoint_keys=["giga-max-sberai","gemini", "giga-max-fin-aifa", "airi-giga"]
test_endpoint_keys_embeddings=["embeddings", "giga-max-fin-aifa"]
test_endpoint_keys_files=["giga-max-fin-aifa", "airi-giga"]
```


2. Run:
- `pytest` :: to run all tests
- `pytest -s` :: to run all tests and show logs
- `pytest --stepwise` :: to stop on first fail
- `pytest -k airi` :: to filter tests which have `airi` as substring
- `pytest -k 'not airi'` :: to filter tests which **have not** `airi` as substring
- `pytest -k airi -k file` :: many filters supported
- `pytest -k aifa -k file --collect-only` :: just show generated filtered tests, without running

Output:
```text
<Dir mmar-llm>
  <Package tests>
    <Module test_get_response.py>
      <Function test_get_response_with_file[giga-max-fin-aifa]>
        <Function test_get_response_with_file[airi-giga]>
```
