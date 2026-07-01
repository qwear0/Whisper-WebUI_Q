# Production Code Review: Whisper-WebUI QSD API and ElevenLabs diff

- Date: 2026-07-01
- Mode: diff
- Repository: `/apps/Whisper-WebUI`
- Inputs: `git diff`, untracked QSD API / ElevenLabs files, targeted tests, Docker targeted tests
- Report path: `tmp/reports/2026-07-01_Review/diff.md`
- Review status: partial: changed-path review and targeted test suites are complete; full repository tests were stopped after no progress in model-heavy transcription tests.

## Executive Summary

- Found and fixed a queue-blocking upload failure path in `QSDTranscriptionService.create_transcription`.
- Found and fixed import boundaries that made the QSD API path depend on UI/ML-only modules before it needed them.
- Found and fixed two integration regressions exposed by container tests: an implicit `torch` dependency and a missing `translation` default config block.
- Verified the QSD API now forces ElevenLabs diarization for QSD-created ElevenLabs jobs by default.
- Targeted local and Docker tests pass for the changed QSD, ElevenLabs, task store, and provider paths.

## Scope and Method

### Reviewed

- `modules/qsd_api/*` - QSD HTTP API, auth, job scheduling, upload handling, status serialization.
- `modules/elevenlabs_transcription/*` - ElevenLabs request settings, chunking, pipeline, and package import behavior.
- `app.py` - provider defaults, ElevenLabs settings, QSD router mounting, and transcription dispatch.
- `modules/utils/task_status_store.py` - task lifecycle fields used by QSD jobs.
- `modules/utils/files_manager.py`, `modules/whisper/base_transcription_pipeline.py`, `modules/whisper/data_classes.py` - changed import/runtime dependencies.
- `configs/default_parameters.yaml`, `docker-compose.yaml`, `requirements.txt`, `README.md` - runtime defaults and deployment contract.
- New and changed tests under `tests/` for QSD API, ElevenLabs, providers, and task status storage.

### Commands and checks

- `git diff --check` - passed.
- `/qsd/.venv/bin/python -m py_compile ...` - passed for changed Python paths.
- `/qsd/.venv/bin/python -m pytest -rs tests/test_qsd_api.py tests/test_elevenlabs_client.py tests/test_task_status_store.py tests/test_elevenlabs_chunker.py` - `25 passed`.
- `docker run --rm -v /apps/Whisper-WebUI:/work -w /work --entrypoint python jhj0517/whisper-webui:latest -m pytest -rs tests/test_qsd_api.py tests/test_elevenlabs_client.py tests/test_task_status_store.py tests/test_elevenlabs_chunker.py tests/test_elevenlabs_segments.py tests/test_elevenlabs_pipeline.py tests/test_transcription_provider.py` - `33 passed, 4 warnings`.
- Full Docker test attempt with `jiwer` installed - stopped after no progress in model-heavy transcription tests; earlier output exposed the `torch` and `translation` regressions fixed here.

### Not reviewed

- Live ElevenLabs network calls with a real API key were not executed.
- Full model inference behavior was not revalidated end to end because the full suite did not complete in this environment.

## Threat and Invariant Model

- Assets and trust boundaries: authenticated QSD clients upload audio and can select provider/output directory; ElevenLabs calls cross an external API boundary; local output paths and task status are persisted on disk.
- Critical invariants checked: API key enforcement, one active QSD transcription at a time, failed uploads must not leave active tasks, provider selection must be recorded, QSD ElevenLabs jobs must use diarization by default, status responses must not expose secret defaults.
- Highest-risk entry points or flows: `POST /qsd/transcriptions`, upload persistence before scheduling, provider dispatch in `App.transcribe_files_by_provider`, default config loading at application startup.

## Findings

### F-001: Upload read failures could leave the QSD worker permanently busy

- Severity: Medium
- Status: Validated
- Area: API Contract
- Files: `modules/qsd_api/service.py`, `tests/test_qsd_api.py`
- Standards mapping: n/a
- Expected invariant: A failed upload must transition the created task to a terminal failed state and release the single-worker gate.
- Evidence: The task was created before `_save_upload()`. If `upload_file.read()` raised after task creation, the exception path did not mark the task terminal, so `list_active_tasks()` could continue returning the orphaned task and future `POST /qsd/transcriptions` calls would receive 409 busy responses.
- Validation: Added `test_qsd_upload_save_error_marks_task_failed_and_unblocks_worker`; local targeted tests passed with `25 passed`, Docker targeted tests passed with `33 passed`.
- Impact: A transient upload/storage error could block all later QSD transcription jobs until manual task cleanup or process restart.
- Recommendation: Keep upload persistence errors inside the task lifecycle, mark the task failed, clean partial upload directories, and close the upload object reliably.
- Minimal fix shape: Implemented in `QSDTranscriptionService.create_transcription()` and `_save_upload()` by catching non-HTTP upload errors, failing the task with `mark_finished=True`, deleting the partial task directory, and closing the upload in `finally`.
- Confidence: High

### F-002: Lightweight QSD imports pulled UI and ML dependencies into API-only test paths

- Severity: Medium
- Status: Validated
- Area: Architecture
- Files: `modules/elevenlabs_transcription/__init__.py`, `app.py`, `modules/utils/files_manager.py`
- Standards mapping: n/a
- Expected invariant: Importing QSD API schemas/services and ElevenLabs value models should not require the Gradio UI or heavy transcription pipeline modules.
- Evidence: Importing `modules.elevenlabs_transcription.models` executed the package `__init__`, which also imported the pipeline and pulled transcription/UI dependencies. Separately, `modules.utils.files_manager` imported `gradio.utils.NamedString` at module import time although QSD only needs media extension constants.
- Validation: The local test environment initially failed on these import paths. After direct submodule imports in `app.py`, a lighter `elevenlabs_transcription.__init__`, and lazy `NamedString` import inside `format_gradio_files()`, `py_compile`, local targeted tests, and Docker targeted tests passed.
- Impact: API-focused tests and deployments could fail before request handling, even when the failing UI/ML dependency was irrelevant to the QSD route being used.
- Recommendation: Preserve the current boundary: model/config imports stay lightweight, while pipeline/UI modules are imported only by runtime code that needs them.
- Minimal fix shape: Implemented by exporting only `ElevenLabsSettings` and `TranscriptionProvider` from `modules.elevenlabs_transcription`, importing `ElevenLabsTranscriptionPipeline` from its concrete submodule, and importing `NamedString` lazily.
- Confidence: High

### F-003: Base transcription pipeline depended on `torch` through an incidental star import

- Severity: Medium
- Status: Validated
- Area: Tests
- Files: `modules/whisper/base_transcription_pipeline.py`, `modules/whisper/data_classes.py`
- Standards mapping: n/a
- Expected invariant: A module that calls `torch` APIs must import `torch` itself rather than relying on unrelated imported modules to leak that name.
- Evidence: Removing the unused `torch` import from `modules/whisper/data_classes.py` exposed `NameError: name 'torch' is not defined` from `modules/whisper/base_transcription_pipeline.py:get_device` in the Docker full-suite attempt.
- Validation: Added explicit `import torch` in `base_transcription_pipeline.py`; `py_compile` and Docker targeted tests passed afterward.
- Impact: Future cleanup of data classes or import ordering could break runtime device detection during application startup.
- Recommendation: Keep direct imports for runtime dependencies used by a module.
- Minimal fix shape: Implemented by adding explicit `import torch` where `torch` is used and leaving `data_classes.py` free of the unused heavy import.
- Confidence: High

### F-004: Default config removal of `translation` broke translation initialization

- Severity: Medium
- Status: Validated
- Area: Operability
- Files: `configs/default_parameters.yaml`
- Standards mapping: n/a
- Expected invariant: `configs/default_parameters.yaml` must contain all top-level sections loaded by application modules, even when QSD defaults are optimized for ElevenLabs transcription.
- Evidence: The Docker full-suite attempt failed with `KeyError: 'translation'` in `modules/translation/translation_base.py` after the diff removed the default `translation` block.
- Validation: Restored the `translation` block in `configs/default_parameters.yaml`; `py_compile` and Docker targeted tests passed afterward.
- Impact: Application startup or translation-related tests could fail even though the change was intended for transcription defaults.
- Recommendation: Keep unrelated default config sections intact when changing transcription/provider defaults.
- Minimal fix shape: Implemented by restoring the existing `translation.deepl`, `translation.nllb`, and `translation.add_timestamp` defaults.
- Confidence: High

## Watchlist / Needs Follow-up

- The QSD API accepts an explicit `output_dir` and resolves it to an absolute local path. This appears intentional for the internal trusted deployment, but it should remain behind the API key and should be revisited before exposing the service outside a trusted network.
- The full repository test suite did not complete because model-heavy transcription tests made no progress in this environment. A separate run with cached models and a longer timeout is still useful before release.
- Live ElevenLabs request behavior was validated with unit tests/mocks only; a smoke test with a real API key would verify provider-side diarization fields and upload limits.

## Positive Notes

- QSD health/status responses avoid exposing Whisper defaults and tokens.
- Provider selection is stored as requested/actual provider, making fallback or routing behavior observable from task status.
- API authentication uses a constant-time comparison for configured keys.
- The single-worker scheduling lock keeps the QSD queue contract simple and testable.

## Suggested Follow-up

- Run the full test suite in the project Docker image with model caches available.
- Add an integration smoke script for one short ElevenLabs diarized transcription when credentials are available.
- Document the trust assumption around client-provided `output_dir` if QSD API clients expand beyond the local automation path.

## Appendix: Commands

```text
git diff --check
Result: passed.

/qsd/.venv/bin/python -m py_compile app.py modules/qsd_api/__init__.py modules/qsd_api/auth.py modules/qsd_api/exceptions.py modules/qsd_api/router.py modules/qsd_api/schemas.py modules/qsd_api/service.py modules/elevenlabs_transcription/__init__.py modules/elevenlabs_transcription/client.py modules/elevenlabs_transcription/chunker.py modules/elevenlabs_transcription/models.py modules/elevenlabs_transcription/pipeline.py modules/elevenlabs_transcription/segments.py modules/utils/files_manager.py modules/utils/filename.py modules/utils/task_status_store.py modules/whisper/base_transcription_pipeline.py modules/whisper/data_classes.py tests/test_qsd_api.py tests/test_elevenlabs_client.py tests/test_elevenlabs_chunker.py tests/test_elevenlabs_segments.py tests/test_elevenlabs_pipeline.py tests/test_transcription_provider.py tests/test_task_status_store.py
Result: passed.

/qsd/.venv/bin/python -m pytest -rs tests/test_qsd_api.py tests/test_elevenlabs_client.py tests/test_task_status_store.py tests/test_elevenlabs_chunker.py
Result: 25 passed in 1.64s.

docker run --rm -v /apps/Whisper-WebUI:/work -w /work --entrypoint python jhj0517/whisper-webui:latest -m pytest -rs tests/test_qsd_api.py tests/test_elevenlabs_client.py tests/test_task_status_store.py tests/test_elevenlabs_chunker.py tests/test_elevenlabs_segments.py tests/test_elevenlabs_pipeline.py tests/test_transcription_provider.py
Result: 33 passed, 4 warnings in 13.77s.

Full Docker test attempt with jiwer installed
Result: stopped after no progress in model-heavy transcription tests; before stopping, the run exposed the fixed `torch` import and `translation` config regressions.
```
