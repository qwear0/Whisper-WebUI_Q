# QSD Transcription API Adapter

This module exposes a narrow JSON API under `/qsd` from the same process that runs
the Gradio UI. It reuses the current `App` instance, saved UI
defaults, output generation, and `TaskStatusStore`.

## Auth

All `/qsd/*` endpoints require:

- env: `QSD_WHISPER_API_KEY`
- header: `X-QSD-Whisper-Key`

If the env var is not configured, the API routes return `503`. Invalid or missing
headers return `401`.

## Endpoints

- `GET /qsd/health`
- `POST /qsd/transcriptions`
- `GET /qsd/transcriptions?status=active&limit=10`
- `GET /qsd/transcriptions/{task_id}`
- `POST /qsd/transcriptions/{task_id}/cancel`

`POST /qsd/transcriptions` accepts one audio upload and optional `output_dir`,
`request_id`, and `provider` form fields. `provider` is `elevenlabs` by default and
also accepts `whisper`. It does not accept model, language, VAD, diarization, or
other provider settings from QSD. The adapter loads saved WebUI defaults, then
enables ElevenLabs speaker diarization for QSD API submissions by default.

When a QSD job explicitly requests ElevenLabs, provider failures mark the task as
`failed` and never fall back to the local Whisper model. Explicit `provider=whisper`
requests remain available as an operator-controlled mode. Status responses expose
`requested_provider` and `actual_provider`; `actual_provider` remains null for
failed jobs.

The first release enforces a single active worker. New submissions return `409`
while any task is `queued`, `in_progress`, or `cancel_requested`.

## Optional Private Dispatch Watchdog

`modules/qsd_api/watchdog.py` is an opt-in poller in the Whisper-WebUI process. It starts only when all three gates are present: `QSD_TRANSCRIPTION_WATCHDOG_ENABLED=true`, `TRANSCRIBE_PROVIDER=elevenlabs`, and a non-empty `QSD_BACKEND_API_KEY`. The default is off; `whisper` mode never starts this watchdog.

The watchdog needs private network access to both `http://qsd_backend:8000/transcription/dispatch-candidates` and the Dagster GraphQL launcher at `http://enigma_dagster_webserver:3000/graphql`. It sends the matching shared backend key as `X-QSD-API-Key`; provision that caller key manually in the ignored external `.env` loaded by this deployment, and configure the matching accepted key in the backend's ignored environment. Do not put key values in this README, and do not expose either connection through a public proxy.

The backend response is a bounded candidate projection, not audio metadata. Candidates are ordered by their next eligible attempt so a repeatedly deferred old request does not permanently hide queued requests. The watchdog launches `telegram_audio_transcription_job` with the attempt generation and `qsd/capacity_lane=general`; ambiguous launch outcomes are reconciled by private Dagster run tags before another launch is attempted.

Credit preflight runs after staging the audio, before creating a task. It inspects the actual duration with `ffprobe` and estimates `ceil(duration_seconds * 4000 / 3600 * 1.2)` credits (4000/hour plus a 20% reserve). If a subscription reports numeric usage/limit and explicitly disallows extension, keys whose remaining workspace credits fall below the estimate are skipped; all such keys return `insufficient_credits` without a task. Missing/invalid keys still return `no_usable_key`; invalid media returns 422 without a credit code. Unknown limits or extendable plans cannot be proved affordable by this estimate. A positive balance or estimated fit is **not** a vendor quote or key-specific allowance; vendor pricing can vary by model, tier, and chunking. Known rejection before acceptance can defer for 24 hours; ambiguous uploads and failures after acceptance require manual reconciliation, never automatic paid retries. Leave the watchdog OFF unless the operator explicitly accepts the residual risk of an insufficient-balance paid submission; this estimate cannot guarantee strict pre-submit affordability.

Rollout status: this implementation is retained but not deployed, and its database schema has not been applied to the live database. The route is verified from source and tests only; live polling/dispatch has not been exercised. This is not a paid smoke-test result.

## Progress

Status responses include both:

- `updated_at`: any task heartbeat or status update.
- `progress_updated_at`: significant numeric progress movement only.

QSD should use `progress_updated_at` for no-progress timeout decisions.

## Cancellation

Cancellation is cooperative and best-effort in this release. Queued tasks become
`cancelled`. Running tasks become `cancel_requested`, and the worker stops at the
next status callback or stage boundary. If an underlying provider call is blocked
before the next callback, the API cannot kill it immediately.

Output files are returned as filesystem paths in `result_files`; this API does not
provide a download endpoint.

## Tests

- `tests/test_qsd_watchdog.py` covers the explicit provider/enablement gates and ambiguous-launch reconciliation.
- `tests/test_elevenlabs_client.py` covers conservative provider preflight, including exhausted non-extendable usage and uncertain outcomes.
