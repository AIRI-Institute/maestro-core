## simple checks
### CLI
#### Get list of models
`make -s models`

**Advice**: by default `OPENAI_API_BASE` is `localhost:40631/v1`. You can specify other `OPENAI_API_BASE` and `OPENAI_API_KEY`:
`OPENAI_API_BASE="https://inference.airi.net:46783/v1" OPENAI_API_KEY=$(cat airi_api_key.txt) make -s models raw=1`

Sample output:
```python
Model(id='Google/Medgemma-27b-it', created=1770981751, object='model', owned_by='vllm', root='/models/Google/Medgemma-27b-it', parent=None, max_model_len=32768, permission=[{'id': 'modelperm-af23d4a41f0ea106', 'object': 'model_permission', 'created': 1770981751, 'allow_create_engine': False, 'allow_sampling': True, 'allow_logprobs': True, 'allow_search_indices': False, 'allow_view': True, 'allow_fine_tuning': False, 'organization': '*', 'group': None, 'is_blocking': False}])
Model(id='Openai/Gpt-oss-120bь', created=1770981751, object='model', owned_by='vllm', root='/models/Openai/Gpt-oss-120b', parent=None, max_model_len=131072, permission=[{'id': 'modelperm-83ef878c6fcfcdca', 'object': 'model_permission', 'created': 1770981751, 'allow_create_engine': False, 'allow_sampling': True, 'allow_logprobs': True, 'allow_search_indices': False, 'allow_view': True, 'allow_fine_tuning': False, 'organization': '*', 'group': None, 'is_blocking': False}])
Model(id='Sber/GigaChat-Max-V2', created=1770981751, object='model', owned_by='vllm', root='/models/Sber/GigaChat-Max-V2', parent=None, max_model_len=16384, permission=[{'id': 'modelperm-bbb326615aebe3e0', 'object': 'model_permission', 'created': 1770981751, 'allow_create_engine': False, 'allow_sampling': True, 'allow_logprobs': True, 'allow_search_indices': False, 'allow_view': True, 'allow_fine_tuning': False, 'organization': '*', 'group': None, 'is_blocking': False}])
```
#### Check models: is available
`make -s hello`

Sample output:
```text
gigachat_sberai: OK: Hello! My name is GigaChat.
gigachat_aifa: OK: Привет! Меня зовут GigaChat.
gigachat_ispran: OK: Привет! Меня зовут GigaChat.
gigachat_airi: OK: Привет! Я — GigaChat, разговорный искусственный интеллект от Сбера. Чем занимаемся?
gigachat_gpt_oss: OK: Hello! I’m ChatGPT, an AI language model created by OpenAI. How can I help you today?
gemini: OK: I am a large language model, trained by Google. I don't have a personal name. You can just call me Google AI or Bard if you like.
deepseek: OK: Hello! I'm DeepSeek Chat, your AI assistant. You can call me DeepSeek or just Chat if you'd like. 😊 How can I help you today?
```

#### Check models: is structured-output works
`make -s structured`

Sample output:
```json
=== gigachat_sberai (structured) ===
{
  "capabilities": [
    "text",
    "audio",
    "images",
    "call additional skills"
  ],
  "language": "English",
  "name": "GigaChat"
}

=== gigachat_aifa (structured) ===
{
  "language": "Russian",
  "name": "GigaChat"
}
...
```
## how to run tests via pytest

1. Setup `llm-hub-openai` server:
Variant 1: start: `make up`, create `.env` before, see example in `.env.example`
Variant 2: forward port to the server with running llm-hub-openai, e.g. `t up 40631 tank`

2. Create `test.env` in current directory. See example in `test.env.example`:
```env
# Test configuration for model checks
# models_check_rest: model name(s) for simple REST API tests (comma-separated)
models_check_rest=deepseek,gigachat_sberai

# models_check_langchain: model name(s) for LangChain/LangGraph tests (comma-separated)
models_check_langchain=deepseek,gigachat_aifa

# models_check_openai: model name(s) for OpenAI library tests (comma-separated)
models_check_openai=deepseek,gigachat_aifa

OPENAI_API_BASE=http://localhost:40631/v1
```

**Advices**:
- you can define `models_check_all` env parameter instead of `models_check_rest`, `models_check_langchain` and `models_check_openai`.
- run `make -s models join=1` to get list of available models.

Output example: `gigachat_sberai,gigachat_aifa,gigachat_ispran,gigachat_airi,gemini,deepseek`.

Then select models you want to check and put them in `test.env`. Or put in env for one-shot test.

3. Run:
- `pytest` :: to run all tests
- `pytest -s` :: to run all tests and show logs
- `pytest --stepwise` :: to stop on first fail
- `models_check_openai=gigachat_airi pytest -k 'test_openai_tool_calling[gigachat_airi]'` -- to start specific test with specific model
- `pytest -k airi` :: to filter tests which have `airi` as substring
- `pytest -k 'not rest` :: to filter tests which **have not** `rest` as substring
- `pytest -k airi -k langchain` :: OR-filters supported
- `models_check_all='gpt_oss_airi' pytest -k='test_openai_tool_calling[gpt_oss_airi]' -s` :: checking specific model and test
- `pytest -k 'gigachat and langchain and structured_output_json_mode'` :: AND-filters supported
- `pytest -k 'gigachat and langchain and structured_output_json_mode' --collect-only` :: AND-filters supported
- `pytest -k airi -k langchain --collect-only` :: just show generated filtered tests, without running

Output example:
```text
<Dir llm-hub-openai>
  <Dir tests>
    <Module test_langchain.py>
      <Function test_complex_structured_output_json_mode[gigachat_aifa]>
```
