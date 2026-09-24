import os
import argparse
import ast
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from fastapi import FastAPI
import gradio as gr
from gradio_i18n import Translate, gettext as _
from dotenv import load_dotenv
import uvicorn
import yaml

from modules.utils.paths import (FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, WHISPER_MODELS_DIR,
                                 INSANELY_FAST_WHISPER_MODELS_DIR, NLLB_MODELS_DIR, DEFAULT_PARAMETERS_CONFIG_PATH,
                                 UVR_MODELS_DIR, I18N_YAML_PATH)
from modules.utils.files_manager import load_yaml, save_yaml, MEDIA_EXTENSION
from modules.whisper.whisper_factory import WhisperFactory
from modules.translation.nllb_inference import NLLBInference
from modules.ui.htmls import *
from modules.utils.cli_manager import str2bool
from modules.utils.youtube_manager import get_ytmetas
from modules.translation.deepl_api import DeepLAPI
from modules.whisper.data_classes import *
from modules.utils.logger import get_logger
from modules.utils.task_status_store import TaskStatusStore
from modules.qsd_api.router import create_qsd_router
from modules.qsd_api.service import QSDTranscriptionService
from modules.qsd_api.watchdog import Watchdog
from modules.elevenlabs_transcription.models import ElevenLabsSettings, TranscriptionProvider
from modules.elevenlabs_transcription.pipeline import ElevenLabsTranscriptionPipeline


logger = get_logger()
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


class App:
    def __init__(self, args):
        self.args = args
        # Check every 1 hour (3600) for cached files and delete them if older than 1 day (86400)
        self.app = gr.Blocks(css=CSS, theme=self.args.theme, delete_cache=(3600, 86400))
        self.whisper_inf = WhisperFactory.create_whisper_inference(
            whisper_type=self.args.whisper_type,
            whisper_model_dir=self.args.whisper_model_dir,
            faster_whisper_model_dir=self.args.faster_whisper_model_dir,
            insanely_fast_whisper_model_dir=self.args.insanely_fast_whisper_model_dir,
            uvr_model_dir=self.args.uvr_model_dir,
            output_dir=self.args.output_dir,
        )
        self.nllb_inf = NLLBInference(
            model_dir=self.args.nllb_model_dir,
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        self.deepl_api = DeepLAPI(
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        self.task_status_store = TaskStatusStore()
        self.elevenlabs_pipeline = ElevenLabsTranscriptionPipeline()
        self.reconcile_interrupted_tasks()
        self.i18n = load_yaml(I18N_YAML_PATH)
        self.default_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        saved_output_dir = self.get_saved_output_dir(self.default_params)
        if saved_output_dir:
            self.apply_runtime_output_dir(saved_output_dir)
        logger.info(f"Use \"{self.args.whisper_type}\" implementation\n"
                    f"Device \"{self.whisper_inf.device}\" is detected")

    def create_pipeline_inputs(self):
        whisper_params = self.default_params["whisper"]
        vad_params = self.default_params["vad"]
        diarization_params = self.default_params["diarization"]
        uvr_params = self.default_params["bgm_separation"]

        with gr.Row():
            with gr.Column():
                dd_model = gr.Dropdown(choices=self.whisper_inf.available_models, value=whisper_params["model_size"],
                                       label=_("Model"), allow_custom_value=True)
            with gr.Column():
                dd_lang = gr.Dropdown(choices=self.whisper_inf.available_langs + [AUTOMATIC_DETECTION],
                                      value=AUTOMATIC_DETECTION if whisper_params["lang"] == AUTOMATIC_DETECTION.unwrap()
                                      else whisper_params["lang"], label=_("Language"))
                cb_translate = gr.Checkbox(value=whisper_params["is_translate"], label=_("Translate to English?"),
                                           interactive=True)

        with gr.Accordion(_("Advanced Parameters"), open=False):
            whisper_inputs = WhisperParams.to_gradio_inputs(defaults=whisper_params, only_advanced=True,
                                                            whisper_type=self.args.whisper_type,
                                                            available_compute_types=self.whisper_inf.available_compute_types,
                                                            compute_type=self.whisper_inf.current_compute_type)

        with gr.Accordion(_("Background Music Remover Filter"), open=False):
            uvr_inputs = BGMSeparationParams.to_gradio_input(defaults=uvr_params,
                                                             available_models=self.whisper_inf.music_separator.available_models,
                                                             available_devices=self.whisper_inf.music_separator.available_devices,
                                                             device=self.whisper_inf.music_separator.device)

        with gr.Accordion(_("Voice Detection Filter"), open=False):
            vad_inputs = VadParams.to_gradio_inputs(defaults=vad_params)

        with gr.Accordion(_("Diarization"), open=False):
            diarization_inputs = DiarizationParams.to_gradio_inputs(defaults=diarization_params,
                                                                    available_devices=self.whisper_inf.diarizer.available_device,
                                                                    device=self.whisper_inf.diarizer.device)

        pipeline_inputs = [dd_model, dd_lang, cb_translate] + whisper_inputs + vad_inputs + diarization_inputs + uvr_inputs

        return pipeline_inputs

    def create_elevenlabs_inputs(self):
        settings = self.get_saved_elevenlabs_settings()
        language = gr.Dropdown(
            choices=["Automatic", "ru", "en", "de", "fr", "es", "it", "pt", "pl", "uk", "tr", "zh", "ja"],
            value=settings.language_code or "Automatic",
            label="Language",
            allow_custom_value=True,
            info="ISO language code; Automatic lets ElevenLabs detect it.",
        )
        with gr.Row():
            diarize = gr.Checkbox(value=settings.diarize, label="Speaker diarization")
            num_speakers = gr.Slider(
                minimum=1,
                maximum=32,
                step=1,
                value=settings.num_speakers or 2,
                label="Maximum speakers",
            )
            tag_audio_events = gr.Checkbox(value=settings.tag_audio_events, label="Tag audio events")
        transcript_mode = gr.Radio(
            choices=["Verbatim", "Clean"],
            value="Clean" if settings.no_verbatim else "Verbatim",
            label="Transcript mode",
        )
        keyterms = gr.Textbox(
            value=", ".join(settings.keyterms),
            label="Keyterms",
            placeholder="Names and domain terms, separated by commas",
        )
        return [language, diarize, num_speakers, tag_audio_events, transcript_mode, keyterms]

    def transcribe_file_with_task_tracking(self,
                                           files=None,
                                           input_folder_path: str | None = None,
                                           include_subdirectory: bool | None = None,
                                           save_same_dir: bool | None = None,
                                           provider: str = "elevenlabs",
                                           file_format: str = "SRT",
                                           add_timestamp: bool = True,
                                           output_dir: str | None = None,
                                           elevenlabs_language: str | None = None,
                                           elevenlabs_diarize: bool = True,
                                           elevenlabs_num_speakers: int | float | None = 2,
                                           elevenlabs_tag_audio_events: bool = False,
                                           elevenlabs_transcript_mode: str = "Verbatim",
                                           elevenlabs_keyterms: str | None = None,
                                           progress=gr.Progress(),
                                           *pipeline_params):
        selected_provider = TranscriptionProvider.parse(provider)
        elevenlabs_settings = ElevenLabsSettings.from_ui(
            elevenlabs_language,
            elevenlabs_diarize,
            elevenlabs_num_speakers,
            elevenlabs_tag_audio_events,
            elevenlabs_transcript_mode,
            elevenlabs_keyterms,
            defaults=self.get_elevenlabs_config_defaults(),
        )
        self.cache_all_transcription_ui_settings(
            selected_provider.value,
            file_format,
            add_timestamp,
            output_dir,
            elevenlabs_settings,
            *pipeline_params,
        )
        label = self.describe_file_source(files=files, input_folder_path=input_folder_path)
        task_id = self.task_status_store.create_task(
            task_type="transcription",
            source_kind="file",
            label=label,
            message="Preparing transcription..",
            requested_provider=selected_provider.value,
        )
        started_at = time.monotonic()
        self.task_status_store.update_task(task_id, status="in_progress", mark_started=True)

        try:
            result_text, result_files, actual_provider = self.transcribe_files_by_provider(
                files=files,
                provider=selected_provider,
                file_format=file_format,
                add_timestamp=add_timestamp,
                output_dir=output_dir,
                progress=progress,
                pipeline_params=list(pipeline_params),
                elevenlabs_settings=elevenlabs_settings,
                status_callback=self.build_status_callback(task_id),
                input_folder_path=input_folder_path,
                include_subdirectory=bool(include_subdirectory),
                save_same_dir=bool(save_same_dir),
            )
        except Exception as error:
            self.task_status_store.update_task(
                task_id,
                status="failed",
                message="Transcription failed.",
                error=str(error),
                duration_seconds=time.monotonic() - started_at,
                mark_finished=True,
            )
            raise

        self.task_status_store.update_task(
            task_id,
            status="completed",
            progress=1.0,
            message="Completed.",
            current_item="",
            result_files=self.normalize_result_files(result_files),
            duration_seconds=time.monotonic() - started_at,
            mark_finished=True,
            actual_provider=actual_provider,
        )
        return result_text, result_files

    def transcribe_files_by_provider(
        self,
        *,
        files,
        provider: TranscriptionProvider | str,
        file_format: str,
        add_timestamp: bool,
        output_dir: str | None,
        progress,
        pipeline_params: list[Any],
        elevenlabs_settings: ElevenLabsSettings,
        status_callback=None,
        input_folder_path: str | None = None,
        include_subdirectory: bool = False,
        save_same_dir: bool = False,
        allow_whisper_fallback: bool = True,
    ) -> tuple[str, list[str], str]:
        selected_provider = provider if isinstance(provider, TranscriptionProvider) else TranscriptionProvider.parse(provider)
        selected_output_dir = self.normalize_output_dir(output_dir) or self.get_default_output_dir()

        if selected_provider is TranscriptionProvider.WHISPER:
            result_text, result_files = self.whisper_inf.transcribe_file(
                files,
                input_folder_path,
                include_subdirectory,
                save_same_dir,
                file_format,
                add_timestamp,
                progress,
                *pipeline_params,
                output_dir=selected_output_dir,
                status_callback=status_callback,
            )
            return result_text, self.normalize_result_files(result_files), "whisper"

        if input_folder_path and input_folder_path.strip():
            raise ValueError("Local folder input is available only with the Whisper provider.")

        def whisper_fallback(source: str, fallback_status_callback):
            result_text, result_files = self.whisper_inf.transcribe_file(
                [source],
                None,
                False,
                False,
                file_format,
                add_timestamp,
                progress,
                *pipeline_params,
                output_dir=selected_output_dir,
                status_callback=fallback_status_callback,
            )
            return result_text, self.normalize_result_files(result_files)

        return self.elevenlabs_pipeline.transcribe_files(
            files,
            settings=elevenlabs_settings,
            file_format=file_format,
            add_timestamp=add_timestamp,
            output_dir=selected_output_dir,
            status_callback=status_callback,
            fallback=whisper_fallback if allow_whisper_fallback else None,
        )

    def transcribe_youtube_with_task_tracking(self,
                                              youtube_link: str,
                                              file_format: str = "SRT",
                                              add_timestamp: bool = True,
                                              progress=gr.Progress(),
                                              *pipeline_params):
        label = youtube_link.strip() if youtube_link else "Youtube task"
        task_id = self.task_status_store.create_task(
            task_type="transcription",
            source_kind="youtube",
            label=label,
            message="Preparing Youtube transcription..",
        )
        started_at = time.monotonic()
        self.task_status_store.update_task(task_id, status="in_progress", mark_started=True)

        try:
            result_text, result_file = self.whisper_inf.transcribe_youtube(
                youtube_link,
                file_format,
                add_timestamp,
                progress,
                *pipeline_params,
                status_callback=self.build_status_callback(task_id),
            )
        except Exception as error:
            self.task_status_store.update_task(
                task_id,
                status="failed",
                message="Youtube transcription failed.",
                error=str(error),
                duration_seconds=time.monotonic() - started_at,
                mark_finished=True,
            )
            raise

        self.task_status_store.update_task(
            task_id,
            status="completed",
            progress=1.0,
            message="Completed.",
            current_item="",
            result_files=self.normalize_result_files(result_file),
            duration_seconds=time.monotonic() - started_at,
            mark_finished=True,
        )
        return result_text, result_file

    def transcribe_mic_with_task_tracking(self,
                                          mic_audio: str,
                                          file_format: str = "SRT",
                                          add_timestamp: bool = True,
                                          progress=gr.Progress(),
                                          *pipeline_params):
        task_id = self.task_status_store.create_task(
            task_type="transcription",
            source_kind="mic",
            label="Microphone recording",
            message="Preparing microphone transcription..",
        )
        started_at = time.monotonic()
        self.task_status_store.update_task(task_id, status="in_progress", mark_started=True)

        try:
            result_text, result_file = self.whisper_inf.transcribe_mic(
                mic_audio,
                file_format,
                add_timestamp,
                progress,
                *pipeline_params,
                status_callback=self.build_status_callback(task_id),
            )
        except Exception as error:
            self.task_status_store.update_task(
                task_id,
                status="failed",
                message="Microphone transcription failed.",
                error=str(error),
                duration_seconds=time.monotonic() - started_at,
                mark_finished=True,
            )
            raise

        self.task_status_store.update_task(
            task_id,
            status="completed",
            progress=1.0,
            message="Completed.",
            current_item="",
            result_files=self.normalize_result_files(result_file),
            duration_seconds=time.monotonic() - started_at,
            mark_finished=True,
        )
        return result_text, result_file

    def build_status_callback(self, task_id: str):
        state = {
            "last_progress": None,
            "last_message": None,
            "last_item": None,
            "last_write_at": 0.0,
        }

        def callback(progress_value: float | None, message: str, current_item: str | None = None):
            now = time.monotonic()
            normalized_progress = None if progress_value is None else round(float(progress_value), 4)
            progress_changed = (
                normalized_progress is not None and (
                    state["last_progress"] is None or
                    abs(normalized_progress - state["last_progress"]) >= 0.01 or
                    normalized_progress >= 1.0
                )
            )
            should_write = (
                message != state["last_message"] or
                current_item != state["last_item"] or
                progress_changed or
                now - state["last_write_at"] >= 2.0
            )
            if not should_write:
                return

            self.task_status_store.update_task(
                task_id,
                status="in_progress",
                progress=normalized_progress if progress_changed else None,
                message=message,
                current_item=current_item,
            )
            if progress_changed:
                state["last_progress"] = normalized_progress
            state["last_message"] = message
            state["last_item"] = current_item
            state["last_write_at"] = now

        return callback

    def reconcile_interrupted_tasks(self) -> None:
        interrupted_count = self.task_status_store.mark_interrupted_tasks()
        if interrupted_count:
            logger.warning(
                f"Marked {interrupted_count} unfinished transcription task(s) as failed after application restart."
            )

    def render_task_monitor_html(self) -> str:
        tasks = self.task_status_store.list_tasks(limit=10)
        active_statuses = {"queued", "in_progress", "cancel_requested"}
        active_tasks = [task for task in tasks if task["status"] in active_statuses]
        recent_tasks = [task for task in tasks if task["status"] not in active_statuses]

        sections = [
            '<div class="task-monitor">',
            self.render_task_group(
                title="Active Tasks",
                tasks=active_tasks,
                empty_message="No active transcription tasks. This panel refreshes automatically.",
                section_kind="active",
            ),
        ]

        if recent_tasks:
            sections.append(
                self.render_task_group(
                    title="Recent Tasks",
                    tasks=recent_tasks,
                    empty_message="",
                    section_kind="recent",
                )
            )

        sections.append("</div>")
        return "".join(sections)

    def render_task_group(
        self,
        title: str,
        tasks: list[dict],
        empty_message: str,
        section_kind: str = "default",
    ) -> str:
        section_class = "task-monitor__section"
        if section_kind:
            section_class += f" task-monitor__section--{section_kind}"

        if not tasks:
            return (
                f'<section class="{section_class}">'
                f'<div class="task-monitor__title">{escape(title)}</div>'
                f'<div class="task-monitor__empty">{escape(empty_message)}</div>'
                "</section>"
            )

        cards = "".join(self.render_task_card(task) for task in tasks)
        return (
            f'<section class="{section_class}">'
            f'<div class="task-monitor__title">{escape(title)}</div>'
            f'<div class="task-monitor__cards">{cards}</div>'
            "</section>"
        )

    def render_task_card(self, task: dict) -> str:
        status = task["status"]
        progress = task.get("progress")
        progress_html = ""
        if progress is not None:
            progress_percent = round(progress * 100)
            progress_html = (
                '<div class="task-monitor__progress">'
                '<div class="task-monitor__progress-track">'
                f'<span class="task-monitor__progress-fill" style="width: {progress_percent}%;"></span>'
                "</div>"
                f'<div class="task-monitor__progress-text">{progress_percent}%</div>'
                "</div>"
            )

        details = []
        if task.get("message"):
            details.append(f'<div class="task-monitor__message">{escape(task["message"])}</div>')

        outputs = task.get("result_files") or []
        if outputs:
            rendered_outputs = ", ".join(escape(Path(item).name) for item in outputs[:3])
            if len(outputs) > 3:
                rendered_outputs += f" (+{len(outputs) - 3} more)"
            details.append(f'<div class="task-monitor__meta">Outputs: {rendered_outputs}</div>')

        if task.get("error"):
            details.append(f'<div class="task-monitor__meta">Error: {escape(task["error"])}</div>')

        details.append(
            f'<div class="task-monitor__meta">Updated {escape(self.format_relative_time(task["updated_at"]))}</div>'
        )
        if task.get("duration_seconds"):
            details.append(
                f'<div class="task-monitor__meta">Duration {escape(self.format_duration(task["duration_seconds"]))}</div>'
            )

        return (
            f'<article class="task-monitor__card task-monitor__card--{escape(status)}">'
            '<div class="task-monitor__card-header">'
            f'<span class="task-monitor__badge task-monitor__badge--{escape(status)}">{escape(status.replace("_", " "))}</span>'
            f'<span class="task-monitor__source">{escape(task["source_kind"])}</span>'
            "</div>"
            f'<div class="task-monitor__label">{escape(task["current_item"] or task["label"])}</div>'
            f"{progress_html}"
            f'{"".join(details)}'
            "</article>"
        )

    @staticmethod
    def describe_file_source(files=None, input_folder_path: str | None = None) -> str:
        if input_folder_path:
            return f"Folder: {Path(input_folder_path).name or input_folder_path}"

        file_names = []
        for file in files or []:
            if hasattr(file, "name"):
                file_names.append(Path(file.name).name)
            else:
                file_names.append(Path(str(file)).name)

        if not file_names:
            return "Uploaded files"
        if len(file_names) == 1:
            return file_names[0]
        return f"{file_names[0]} (+{len(file_names) - 1} more)"

    @staticmethod
    def normalize_result_files(result_files) -> list[str]:
        if result_files is None:
            return []
        if isinstance(result_files, list):
            return [str(item) for item in result_files]
        return [str(result_files)]

    def cache_transcription_settings(
        self,
        file_format: str = "SRT",
        add_timestamp: bool = True,
        *pipeline_params,
    ) -> None:
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            params = self.whisper_inf.validate_gradio_values(params)
            self.whisper_inf.cache_parameters(
                params=params,
                file_format=file_format,
                add_timestamp=add_timestamp,
            )
            self.default_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH) or self.default_params
        except Exception as error:
            logger.warning(f"Failed to save transcription settings: {error}")

    def cache_transcription_ui_settings(
        self,
        file_format: str = "SRT",
        add_timestamp: bool = True,
        output_dir: str | None = None,
        *pipeline_params,
    ) -> None:
        self.cache_output_dir(output_dir)
        self.cache_transcription_settings(file_format, add_timestamp, *pipeline_params)

    def cache_all_transcription_ui_settings(
        self,
        provider: str,
        file_format: str,
        add_timestamp: bool,
        output_dir: str | None,
        elevenlabs_settings: ElevenLabsSettings,
        *pipeline_params,
    ) -> None:
        self.cache_output_dir(output_dir)
        self.cache_transcription_settings(file_format, add_timestamp, *pipeline_params)
        self.cache_provider_settings(provider, elevenlabs_settings)

    def cache_transcription_controls(
        self,
        provider: str,
        file_format: str,
        add_timestamp: bool,
        output_dir: str | None,
        language_code: str | None,
        diarize: bool,
        num_speakers: int | float | None,
        tag_audio_events: bool,
        transcript_mode: str,
        keyterms: str | None,
        *pipeline_params,
    ) -> None:
        settings = ElevenLabsSettings.from_ui(
            language_code,
            diarize,
            num_speakers,
            tag_audio_events,
            transcript_mode,
            keyterms,
            defaults=self.get_elevenlabs_config_defaults(),
        )
        self.cache_all_transcription_ui_settings(
            provider,
            file_format,
            add_timestamp,
            output_dir,
            settings,
            *pipeline_params,
        )

    def cache_provider_settings(
        self,
        provider: str,
        settings: ElevenLabsSettings | None = None,
    ) -> None:
        try:
            selected = TranscriptionProvider.parse(provider)
            cached_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH) or {}
            transcription = cached_params.get("transcription")
            if not isinstance(transcription, dict):
                transcription = {}
            transcription["provider"] = selected.value
            cached_params["transcription"] = transcription
            if settings is not None:
                cached_params["elevenlabs"] = settings.to_cache()
            save_yaml(cached_params, DEFAULT_PARAMETERS_CONFIG_PATH)
            self.default_params = cached_params
        except Exception as error:
            logger.warning(f"Failed to save transcription provider settings: {error}")

    def select_transcription_provider(self, provider: str):
        selected = TranscriptionProvider.parse(provider)
        self.cache_provider_settings(selected.value)
        return (
            selected.value,
            gr.Button(variant="primary" if selected is TranscriptionProvider.ELEVENLABS else "secondary"),
            gr.Button(variant="primary" if selected is TranscriptionProvider.WHISPER else "secondary"),
            gr.Tabs(selected=selected.value),
        )

    def cache_output_dir(self, output_dir: str | None) -> None:
        try:
            cached_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH) or {}
            selected_output_dir = self.normalize_output_dir(output_dir)

            if selected_output_dir:
                cached_params["output_dir"] = selected_output_dir
                self.apply_runtime_output_dir(selected_output_dir)
            else:
                cached_params.pop("output_dir", None)
                self.apply_runtime_output_dir(str(self.args.output_dir))

            save_yaml(cached_params, DEFAULT_PARAMETERS_CONFIG_PATH)
            self.default_params = cached_params
        except Exception as error:
            logger.warning(f"Failed to save output path: {error}")

    def get_default_output_dir(self) -> str:
        defaults = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH) or self.default_params
        saved_output_dir = self.get_saved_output_dir(defaults)
        if saved_output_dir:
            self.apply_runtime_output_dir(saved_output_dir)
            return saved_output_dir

        fallback_output_dir = self.normalize_output_dir(getattr(self.whisper_inf, "output_dir", self.args.output_dir))
        return str(fallback_output_dir or self.args.output_dir)

    @staticmethod
    def get_saved_output_dir(defaults: Any) -> str | None:
        if not isinstance(defaults, dict):
            return None

        raw_output_dir = defaults.get("output_dir")
        if not isinstance(raw_output_dir, str):
            return None

        return App.normalize_output_dir(raw_output_dir)

    @staticmethod
    def normalize_output_dir(output_dir: str | None) -> str | None:
        if output_dir is None:
            return None

        output_dir = str(output_dir).strip()
        if not output_dir:
            return None

        return os.path.abspath(os.path.expanduser(output_dir))

    def apply_runtime_output_dir(self, output_dir: str) -> None:
        self.whisper_inf.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        music_separator = getattr(self.whisper_inf, "music_separator", None)
        if music_separator is not None:
            music_separator.output_dir = os.path.join(output_dir, "UVR")
            os.makedirs(os.path.join(music_separator.output_dir, "instrumental"), exist_ok=True)
            os.makedirs(os.path.join(music_separator.output_dir, "vocals"), exist_ok=True)

    def load_transcription_ui_settings(self) -> list[Any]:
        file_format, add_timestamp, pipeline_params = self.get_saved_transcription_defaults()
        provider = self.get_saved_transcription_provider().value
        elevenlabs_values = self.get_saved_elevenlabs_settings().to_ui()
        return [provider, file_format, add_timestamp, self.get_default_output_dir(), *elevenlabs_values, *pipeline_params]

    def get_saved_transcription_provider(self) -> TranscriptionProvider:
        defaults = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH) or self.default_params or {}
        transcription = defaults.get("transcription") if isinstance(defaults, dict) else None
        raw_provider = transcription.get("provider") if isinstance(transcription, dict) else None
        try:
            return TranscriptionProvider.parse(raw_provider)
        except ValueError:
            return TranscriptionProvider.ELEVENLABS

    def get_elevenlabs_config_defaults(self) -> dict[str, Any]:
        defaults = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH) or self.default_params or {}
        values = defaults.get("elevenlabs") if isinstance(defaults, dict) else None
        return values if isinstance(values, dict) else {}

    def get_saved_elevenlabs_settings(self) -> ElevenLabsSettings:
        return ElevenLabsSettings.from_cache(self.get_elevenlabs_config_defaults())

    def get_saved_transcription_defaults(self) -> tuple[str, bool, list[Any]]:
        defaults = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH) or {}
        whisper_defaults = self.select_model_values(WhisperParams, defaults.get("whisper"))
        vad_defaults = self.select_model_values(VadParams, defaults.get("vad"))
        diarization_defaults = self.select_model_values(DiarizationParams, defaults.get("diarization"))
        bgm_defaults = self.select_model_values(BGMSeparationParams, defaults.get("bgm_separation"))

        params = TranscriptionPipelineParams(
            whisper=WhisperParams(**whisper_defaults),
            vad=VadParams(**vad_defaults),
            diarization=DiarizationParams(**diarization_defaults),
            bgm_separation=BGMSeparationParams(**bgm_defaults),
        )
        whisper_section = defaults.get("whisper") if isinstance(defaults.get("whisper"), dict) else {}
        file_format = whisper_section.get("file_format", "SRT")
        add_timestamp = bool(whisper_section.get("add_timestamp", True))
        return file_format, add_timestamp, params.to_list()

    @staticmethod
    def select_model_values(model_class, values: Any) -> dict[str, Any]:
        if not isinstance(values, dict):
            return {}
        return {
            field_name: values[field_name]
            for field_name in model_class.model_fields
            if field_name in values
        }

    @staticmethod
    def format_relative_time(value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return value

        delta_seconds = max(0, int((datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()))
        if delta_seconds < 5:
            return "just now"
        if delta_seconds < 60:
            return f"{delta_seconds}s ago"

        minutes, seconds = divmod(delta_seconds, 60)
        if minutes < 60:
            return f"{minutes}m {seconds}s ago"

        hours, minutes = divmod(minutes, 60)
        if hours < 24:
            return f"{hours}h {minutes}m ago"

        days, hours = divmod(hours, 24)
        return f"{days}d {hours}h ago"

    @staticmethod
    def format_duration(duration_seconds: float) -> str:
        total_seconds = max(0, int(round(duration_seconds)))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)

        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def launch(self):
        with self.app:
            lang = gr.Radio(choices=list(self.i18n.keys()),
                            label=_("Language"), interactive=True,
                            visible=False,  # Set it by development purpose.
                            )
            with Translate(self.i18n):  # Add `lang = lang` here to test dynamic change of the languages.
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(MARKDOWN, elem_id="md_project")
                        with gr.Accordion("Task Monitor", open=True):
                            task_monitor = gr.HTML()
                            task_monitor_refresh = gr.Timer(value=2, active=True)
                with gr.Column():
                    with gr.Row(visible=False):
                        tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                        files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3, interactive=False)
                        btn_openfolder = gr.Button('📂', scale=1)

                    input_file = gr.Files(type="filepath", label=_("Upload File here"), file_types=MEDIA_EXTENSION)

                    with gr.Row():
                        btn_run = gr.Button("Транскрибация", variant="primary")

                    initial_provider = self.get_saved_transcription_provider()
                    provider_state = gr.State(initial_provider.value)
                    with gr.Row():
                        btn_elevenlabs = gr.Button(
                            "ElevenLabs",
                            variant="primary" if initial_provider is TranscriptionProvider.ELEVENLABS else "secondary",
                        )
                        btn_whisper = gr.Button(
                            "Whisper",
                            variant="primary" if initial_provider is TranscriptionProvider.WHISPER else "secondary",
                        )

                    with gr.Tabs(selected=initial_provider.value) as provider_tabs:
                        with gr.Tab("ElevenLabs", id="elevenlabs", interactive=False):
                            elevenlabs_inputs = self.create_elevenlabs_inputs()
                        with gr.Tab("Whisper", id="whisper", interactive=False):
                            tb_input_folder = gr.Textbox(
                                label="Input Folder Path (Optional)",
                                info="Local folder inputs are supported only by Whisper. Leave empty when uploading files.",
                                visible=self.args.colab,
                                value="",
                            )
                            cb_include_subdirectory = gr.Checkbox(
                                label="Include Subdirectory Files",
                                visible=self.args.colab,
                                value=False,
                            )
                            cb_save_same_dir = gr.Checkbox(
                                label="Save outputs at same directory",
                                visible=self.args.colab,
                                value=True,
                            )
                            pipeline_params = self.create_pipeline_inputs()

                    with gr.Row():
                        dd_file_format = gr.Dropdown(
                            choices=["SRT", "WebVTT", "txt", "LRC"],
                            value=self.default_params["whisper"]["file_format"],
                            label=_("File Format"),
                        )
                        cb_timestamp = gr.Checkbox(
                            value=bool(self.default_params["whisper"]["add_timestamp"]),
                            label=_("Add a timestamp to the end of the filename"),
                            interactive=True,
                        )
                        tb_output_dir = gr.Textbox(
                            label="Output Path",
                            value=self.get_default_output_dir(),
                            placeholder=self.args.output_dir,
                        )

                    btn_elevenlabs.click(
                        fn=lambda: self.select_transcription_provider("elevenlabs"),
                        inputs=None,
                        outputs=[provider_state, btn_elevenlabs, btn_whisper, provider_tabs],
                        queue=False,
                        show_progress="hidden",
                    )
                    btn_whisper.click(
                        fn=lambda: self.select_transcription_provider("whisper"),
                        inputs=None,
                        outputs=[provider_state, btn_elevenlabs, btn_whisper, provider_tabs],
                        queue=False,
                        show_progress="hidden",
                    )
                    params = [
                        input_file,
                        tb_input_folder,
                        cb_include_subdirectory,
                        cb_save_same_dir,
                        provider_state,
                        dd_file_format,
                        cb_timestamp,
                        tb_output_dir,
                        *elevenlabs_inputs,
                        *pipeline_params,
                    ]
                    settings_params = [
                        provider_state,
                        dd_file_format,
                        cb_timestamp,
                        tb_output_dir,
                        *elevenlabs_inputs,
                        *pipeline_params,
                    ]
                    for setting_input in settings_params:
                        setting_input.change(
                            fn=self.cache_transcription_controls,
                            inputs=settings_params,
                            outputs=None,
                            queue=False,
                            show_progress="hidden",
                        )

                    btn_run.click(fn=self.transcribe_file_with_task_tracking,
                                  inputs=params,
                                  outputs=[tb_indicator, files_subtitles])
                    btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)

            self.app.load(
                fn=self.load_transcription_ui_settings,
                inputs=None,
                outputs=settings_params,
                queue=False,
                show_progress="hidden",
            )
            self.app.load(
                fn=self.render_task_monitor_html,
                inputs=None,
                outputs=task_monitor,
                queue=False,
                show_progress="hidden",
            )
            task_monitor_refresh.tick(
                fn=self.render_task_monitor_html,
                inputs=None,
                outputs=task_monitor,
                queue=False,
                show_progress="hidden",
            )

        # Launch the app with optional gradio settings
        self.launch_runtime()

    def launch_runtime(self):
        if self.args.legacy_gradio_launch:
            self.launch_gradio_blocks()
            return

        parent_app = self.create_parent_app()
        uvicorn.run(
            parent_app,
            host=self.args.server_name or "127.0.0.1",
            port=self.args.server_port or 7860,
            ssl_keyfile=self.args.ssl_keyfile,
            ssl_certfile=self.args.ssl_certfile,
            ssl_keyfile_password=self.args.ssl_keyfile_password,
        )

    def launch_gradio_blocks(self):
        args = self.args
        self.app.queue(
            api_open=args.api_open
        ).launch(
            share=args.share,
            server_name=args.server_name,
            server_port=args.server_port,
            auth=(args.username, args.password) if args.username and args.password else None,
            root_path=args.root_path,
            inbrowser=args.inbrowser,
            ssl_verify=args.ssl_verify,
            ssl_keyfile=args.ssl_keyfile,
            ssl_keyfile_password=args.ssl_keyfile_password,
            ssl_certfile=args.ssl_certfile,
            allowed_paths=self.parse_allowed_paths(args.allowed_paths)
        )

    def create_parent_app(self) -> FastAPI:
        args = self.args
        self.app.queue(api_open=args.api_open)
        parent_app = FastAPI(
            title="Whisper-WebUI",
            description="Whisper-WebUI with QSD transcription API adapter.",
            root_path=args.root_path or "",
        )
        qsd_service = QSDTranscriptionService(self)
        parent_app.include_router(create_qsd_router(qsd_service))
        parent_app.add_event_handler("shutdown", qsd_service.shutdown)
        if (os.environ.get("QSD_TRANSCRIPTION_WATCHDOG_ENABLED") == "true"
                and os.environ.get("TRANSCRIBE_PROVIDER") == "elevenlabs"
                and os.environ.get("QSD_BACKEND_API_KEY", "").strip()):
            watchdog = Watchdog(self.task_status_store)
            parent_app.add_event_handler("startup", watchdog.start)
            parent_app.add_event_handler("shutdown", watchdog.stop)

        if args.share:
            logger.warning("Gradio share=True is ignored in FastAPI parent app mode.")
        if args.inbrowser:
            logger.info("Gradio inbrowser=True is ignored in FastAPI parent app mode.")

        gr.mount_gradio_app(
            parent_app,
            self.app,
            path="/",
            server_name=args.server_name or "0.0.0.0",
            server_port=args.server_port or 7860,
            show_api=args.api_open,
            auth=(args.username, args.password) if args.username and args.password else None,
            root_path=args.root_path,
            allowed_paths=self.parse_allowed_paths(args.allowed_paths),
        )
        return parent_app

    @staticmethod
    def parse_allowed_paths(raw_value: str | None) -> list[str] | None:
        if not raw_value:
            return None

        parsed = ast.literal_eval(raw_value)
        if isinstance(parsed, str):
            return [parsed]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        raise ValueError("--allowed_paths must be a string or a list of strings")

    @staticmethod
    def open_folder(folder_path: str):
        if os.path.exists(folder_path):
            os.system(f"start {folder_path}")
        else:
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"The directory path {folder_path} has newly created.")


parser = argparse.ArgumentParser()
parser.add_argument('--whisper_type', type=str, default=WhisperImpl.FASTER_WHISPER.value,
                    choices=[item.value for item in WhisperImpl],
                    help='A type of the whisper implementation (Github repo name)')
parser.add_argument('--share', type=str2bool, default=False, nargs='?', const=True, help='Gradio share value')
parser.add_argument('--server_name', type=str, default=None, help='Gradio server host')
parser.add_argument('--server_port', type=int, default=None, help='Gradio server port')
parser.add_argument('--root_path', type=str, default=None, help='Gradio root path')
parser.add_argument('--username', type=str, default=None, help='Gradio authentication username')
parser.add_argument('--password', type=str, default=None, help='Gradio authentication password')
parser.add_argument('--theme', type=str, default=None, help='Gradio Blocks theme')
parser.add_argument('--colab', type=str2bool, default=False, nargs='?', const=True, help='Is colab user or not')
parser.add_argument('--api_open', type=str2bool, default=False, nargs='?', const=True,
                    help='Enable api or not in Gradio')
parser.add_argument('--legacy_gradio_launch', type=str2bool, default=False, nargs='?', const=True,
                    help='Use Blocks.launch() instead of the FastAPI parent app')
parser.add_argument('--allowed_paths', type=str, default=None, help='Gradio allowed paths')
parser.add_argument('--inbrowser', type=str2bool, default=True, nargs='?', const=True,
                    help='Whether to automatically start Gradio app or not')
parser.add_argument('--ssl_verify', type=str2bool, default=True, nargs='?', const=True,
                    help='Whether to verify SSL or not')
parser.add_argument('--ssl_keyfile', type=str, default=None, help='SSL Key file location')
parser.add_argument('--ssl_keyfile_password', type=str, default=None, help='SSL Key file password')
parser.add_argument('--ssl_certfile', type=str, default=None, help='SSL cert file location')
parser.add_argument('--whisper_model_dir', type=str, default=WHISPER_MODELS_DIR,
                    help='Directory path of the whisper model')
parser.add_argument('--faster_whisper_model_dir', type=str, default=FASTER_WHISPER_MODELS_DIR,
                    help='Directory path of the faster-whisper model')
parser.add_argument('--insanely_fast_whisper_model_dir', type=str,
                    default=INSANELY_FAST_WHISPER_MODELS_DIR,
                    help='Directory path of the insanely-fast-whisper model')
parser.add_argument('--diarization_model_dir', type=str, default=DIARIZATION_MODELS_DIR,
                    help='Directory path of the diarization model')
parser.add_argument('--nllb_model_dir', type=str, default=NLLB_MODELS_DIR,
                    help='Directory path of the Facebook NLLB model')
parser.add_argument('--uvr_model_dir', type=str, default=UVR_MODELS_DIR,
                    help='Directory path of the UVR model')
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory path of the outputs')
if __name__ == "__main__":
    _args = parser.parse_args()
    app = App(args=_args)
    app.launch()
