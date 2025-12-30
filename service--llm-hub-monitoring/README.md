# LLM Hub Monitoring
## Description
**LLM Hub Monitoring** is a services which proxies all requests to **LLM Hub** and forwards it to **LangFuse**.

The key concept: **LLMHub** and **LLMHub-user** (e.g. financier-extractor, financier-reasoner, etc. ) **should not know about tracing at all**. Assumed that it prevents readability degradation.
## Responsible
@tagin
## Problems and assumed solutions
- **problem** :: hard to distinguish different calls
  - **solution** :: add to `mmar_mapi.services.LLMCallProps` some **metadata** in the future

- **problem** :: extra layer of serialization/deserialization can affect performance, especially if there is a lot of data
  - **solution** :: bind **LLM Hub Monitoring** to **LLM Hub** in the environment tangled to the original llm-hub

- **problem** :: LangFuse looks kinda monstrous for the start
  - **solution** :: support some more simple platform
