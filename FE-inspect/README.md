# FE-inspect

Read-only inspector served by the core at `/inspect/`. Run inputs and interaction belong to the separate `/test/` frontend.

- Flow map uses only persisted spans and `parent_span_id`. No sample agents, timing, or fabricated pipeline nodes.
- Waterfall displays recorded start/end times; pending tasks retain unknown start/end. A running timer is calculated from a recorded `started_at` and marked `↻`.
- Span details expose input, output, attributes, errors, agent instance IDs and lifetime scope. Instance `Observed` is a registry observation; `Released` is a registry release, not constructor/destructor timing.
- Existing runs without lifecycle tracing explicitly show missing or partial coverage.
- Polling `/trace` does not acquire the candidate WebSocket lease. UI selection, collapsed nodes and scroll positions survive polling.
- Run search is backed by `/runs?q=...`; older runs and events can be loaded through API pagination.
- Artifacts are read from the backend. Preview failures and backend truncation markers remain visible.
- Interface language is selectable in the header (Vietnamese by default, or English), persisted in `core-studio-locale` and shared with FE-test. Backend content, identifiers and agent names retain their original language. Interview language is selected in FE-test independently.
- On desktop the run list, trace pane and detail pane scroll inside a fixed viewport. Smaller screens stack bounded panes and retain their independent scroll areas.
- Waterfall supports fit view, zoom buttons and a logarithmic 1–64× slider. The time axis scrolls horizontally and can be dragged on its background; agent names and the time header remain sticky. Zoom, selection and scroll positions survive polling.
- Structured input/output, events and JSON artifacts offer an expandable, syntax-colored tree and the complete backend source view. JSON tree rendering is bounded to 2,500 nodes and 13 levels; source view remains available for deeper or larger records.
- Markdown artifacts render a safe common subset: headings, paragraphs, lists, quotes, fences, tables, inline code, emphasis and HTTP(S) links. Raw HTML and unsupported URL schemes render as text. It is not a full CommonMark renderer; use Source for exact original content.
- Backend URL is shared with FE-test through `core-studio-base-url`; the service token uses the tab's `core-studio-token` session storage. The default is the current origin with an optional development token supplied by core.

This frontend has no package dependencies and no run database in browser memory. Its JavaScript state is a view of backend records; reload reads the persisted records again. Persistence is the responsibility of the core's inspector store.
