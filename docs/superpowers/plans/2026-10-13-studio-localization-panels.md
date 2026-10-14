# Core Studio localization and inspection panels

**Goal:** Make both real-data frontends comfortable in Vietnamese and English, with durable per-session interview language and bounded scrolling.

**Architecture:** Keep the existing dependency-free static frontends and core API. UI locale is a browser preference (`core-studio-locale`); interview language is a persisted session input (`interview_language`, vi/en). Render stored content safely without inventing values or modifying artifacts.

**Decisions:** Default Vietnamese; English selectable. Preserve original content and identifiers. Desktop layout stays inside the viewport with independent run, visual, and detail scrolling. Mobile stacks bounded panels. Waterfall uses a separate zoom value and horizontally scrollable timeline with sticky names. JSON and Markdown offer rendered and source modes, escaping HTML and blocking unsafe links. Colors identify span kind and status with accompanying text.

## Delivery lanes

- [x] FE-inspect: locale selector/dictionaries; constrained panels; timeline zoom/pan; safe structured previews; accessible colors.
- [x] FE-test: locale selector/dictionaries; interview-language form field and persisted display; constrained panels; readable report previews.
- [x] Core: validated language contract, stored form input, language guidance in actual profiling and interviewer/followup paths.
- [x] Integration: review translations, alignment of contract and UI, asset packaging, syntax checks and Docker image build.
- [x] Deployment: refresh main stack after verifying current session state, preserve existing volumes; review existing real archived data in browser.

## Evidence and limits

Use syntax/compile checks and Docker build plus visual review of already persisted data. Do not add or run automated tests unless requested. Do not create seeded runs or invoke the configured model just to populate views. Validate layout, locale switching and timeline controls using existing trace coverage. If archived data lacks timing, show the actual missing-timing state.

## Completion

Record changed files, build result, reviewed states and any unverified model-language behavior. No commits or new dependencies are needed.

## Delivery evidence

- JavaScript syntax checks passed for both apps, both dictionaries and the inspector viewer helper. Python compile checks passed for language contract/prompt integrations; whitespace diff check passed.
- Docker app image built and the main container was updated with existing volumes. App and Redis healthy; `/health/ready` reports `ready`. Live OpenAPI advertises language enum `vi,en` and default `vi`.
- Existing real cancelled session `sess_8959acef9e0f` inspected: 67 spans, 9 observed instances, 74 events, 19 artifacts. No new session or fabricated record created.
- At desktop viewport 1652x927, outer document height is 927 for both apps. Inspector trace pane has 6223px of content inside its own scroll; detail pane and FE-test conversation/details scroll independently.
- Waterfall zoom 800% produces 6093px horizontal content; dragging changes scrollLeft to260 while sticky name column remains at x270 and outer page scrollTop stays0.
- Both interface selectors switch Vietnamese/English; create form interview language stays independent. JSON Source renders stored3576 characters; Markdown brief renders9 headings from the stored artifact. Archive text and IDs stay unchanged.
- Screenshots: `.omx/artifacts/studio-ux/waterfall-vi.jpg`, `test-vi.jpg`.
- Asset release queries and missing-helper bootstrap recovery cover old cached HTML during the localization upgrade.
- Automated tests and fresh model runs were not executed. Provider compliance with vi/en prompts, final report locale and narrow/mobile layouts have not been validated live in this change. Markdown is a safe common-syntax subset; structured preview limits are explicit with Source access.
