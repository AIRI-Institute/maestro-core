# mmar-llm

Library to access different LLM's via common API:
- GigaChat
- OpenRouter
- ..

## Usage
### Create `llm_config.json` with llm-endpoints:
```js
{
  "default_endpoint_key": "endpoint_key",
  "warmup": true,
  "wait_seconds_on_llm_retry": [1, 2, 4, 4, 4],
  "endpoints": [
    {
      "key": "endpoint_key",
      "caption": "GigaChat MAX 2",
      "descriptor": "gigachat",
      "args": {
		"authorization_key": "MDAwMDAwMDAtMDAwMC0wMDAwLTAwMDAtMDAwMDAwMDAwMDAwOjAwMDAwMDAwLTAwMDAtMDAwMC0wMDAwLTAwMDAwMDAwMDAwMA=="
      }
    },
	...
  ]
}
```
### Create llm-hub:
```python
from pathlib import Path
from types import SimpleNamespace

from mmar_llm import LLMConfig, LLMHub


def create_llm_hub(llm_config_path: str, tmp_path: str | None = None):
    llm_config = LLMConfig.model_validate_json(Path(llm_config_path).read_text())
    llm_hub_config = SimpleNamespace(llm=llm_config, files_dir=tmp_path)
    llm_hub = LLMHub(llm_hub_config)
    return llm_hub

lh = create_llm_hub('/path/to/llm_config.json')
print(lh.get_response(request='What is your name?'))
```
