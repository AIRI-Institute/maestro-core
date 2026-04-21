# mmar-mimpl

MAESTRO Implementation details:
- `trace_id` :: to pass `trace_id` between services
- `SettingsModel` :: base class for settings in MAESTRO components
- `ResourcesModel` :: base class for models with resources in MAESTRO components
- `parallel_map_ext` :: like `parallel_map`, but respects `TRACE_ID_VAR`
