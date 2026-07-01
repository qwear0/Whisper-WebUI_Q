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
Status responses expose `requested_provider` and `actual_provider`; the latter is
`whisper_fallback` when ElevenLabs failed and the complete source was rerun locally.

The first release enforces a single active worker. New submissions return `409`
while any task is `queued`, `in_progress`, or `cancel_requested`.

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
