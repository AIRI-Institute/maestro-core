# News
## Version 0.2.0 ( 2026-04-21 )
- **BREAKING**: updated **service** `llm-hub`:
  - Migrated to OpenAI-compatible API architecture
  - Replaced JSON configuration with TOML format (`llm-config.toml`)
  - Unified API for multiple providers: OpenAI, Anthropic, GigaChat, DeepSeek, OpenRouter
  - See `service--llm-hub/llm-config.toml.example` for migration guide
- updated **library** `mmar-carl`: see [./lib--mar-carl/RELEASE.md](./lib--mar-carl/RELEASE.md) for details
- removed **service** `llm-hub-monitoring`: LangFuse has native OpenAI API integrations
- other minor fixes and improvements
## Version 0.1.2 ( 2026-02-03 )
- fixed **bugs**: 
  - `question-detector`: added absent model binary
  - `llm-hub`: fixed example
  - `chat-manager-examples`: fixed tracks
- minor updated in `mmar-carl`
## Version 0.1.1 ( 2025-12-30 )
- added **service** `llm-hub-monitoring` -- transparent proxy service implementing `LLMHubAPI` interface, forwarding all requests to the real LLM Hub with Langfuse integration. See example configuration in `maestro-core/service--llm-hub-monitoring/langfuse/`.
- added **service** `frontend-telegram` -- service to integrate MAESTRO with Telegram
- updated **service** `chat-manager-examples` -- added `slot-filling-track` - track for filling slots via conversational form. See example in `service--chat-manager-examples/src/chat_manager_examples/tracks/filling_user_profile.py`.
- minor refinements and fixes
## Version 0.1.0 ( 2025-12-03 )
- added service `document-extractor`
- refined modules structure
- refined LLM classes names:
  - `LLMAccessor` -> `LLMHub`,
  - `Entrypoint` -> `LLMEndpoint`,
  - `entrypoints.json` -> `llm_config.json`
- other minor fixes and improvements
