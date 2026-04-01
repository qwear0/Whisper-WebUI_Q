import os
import argparse
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import gradio as gr
from gradio_i18n import Translate, gettext as _
import yaml

from modules.utils.paths import (FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, WHISPER_MODELS_DIR,
                                 INSANELY_FAST_WHISPER_MODELS_DIR, NLLB_MODELS_DIR, DEFAULT_PARAMETERS_CONFIG_PATH,
                                 UVR_MODELS_DIR, I18N_YAML_PATH)
from modules.utils.files_manager import load_yaml, MEDIA_EXTENSION
from modules.whisper.whisper_factory import WhisperFactory
from modules.translation.nllb_inference import NLLBInference
from modules.ui.htmls import *
from modules.utils.cli_manager import str2bool
from modules.utils.youtube_manager import get_ytmetas
from modules.translation.deepl_api import DeepLAPI
from modules.whisper.data_classes import *
from modules.utils.logger import get_logger
from modules.utils.task_status_store import TaskStatusStore


logger = get_logger()


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
        self.i18n = load_yaml(I18N_YAML_PATH)
        self.default_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        logger.info(f"Use \"{self.args.whisper_type}\" implementation\n"
                    f"Device \"{self.whisper_inf.device}\" is detected")

    def create_pipeline_inputs(self):
        whisper_params = self.default_params["whisper"]
        vad_params = self.default_params["vad"]
        diarization_params = self.default_params["diarization"]
        uvr_params = self.default_params["bgm_separation"]

        with gr.Row():
            dd_model = gr.Dropdown(choices=self.whisper_inf.available_models, value=whisper_params["model_size"],
                                   label=_("Model"), allow_custom_value=True)
            dd_lang = gr.Dropdown(choices=self.whisper_inf.available_langs + [AUTOMATIC_DETECTION],
                                  value=AUTOMATIC_DETECTION if whisper_params["lang"] == AUTOMATIC_DETECTION.unwrap()
                                  else whisper_params["lang"], label=_("Language"))
            dd_file_format = gr.Dropdown(choices=["SRT", "WebVTT", "txt", "LRC"], value=whisper_params["file_format"], label=_("File Format"))
        with gr.Row():
            cb_translate = gr.Checkbox(value=whisper_params["is_translate"], label=_("Translate to English?"),
                                       interactive=True)
        with gr.Row():
            cb_timestamp = gr.Checkbox(value=whisper_params["add_timestamp"],
                                       label=_("Add a timestamp to the end of the filename"),
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

        return (
            pipeline_inputs,
            dd_file_format,
            cb_timestamp
        )

    def transcribe_file_with_task_tracking(self,
                                           files=None,
                                           input_folder_path: str | None = None,
                                           include_subdirectory: bool | None = None,
                                           save_same_dir: bool | None = None,
                                           file_format: str = "SRT",
                                           add_timestamp: bool = True,
                                           progress=gr.Progress(),
                                           *pipeline_params):
        label = self.describe_file_source(files=files, input_folder_path=input_folder_path)
        task_id = self.task_status_store.create_task(
            task_type="transcription",
            source_kind="file",
            label=label,
            message="Preparing transcription..",
        )
        started_at = time.monotonic()
        self.task_status_store.update_task(task_id, status="in_progress")

        try:
            result_text, result_files = self.whisper_inf.transcribe_file(
                files,
                input_folder_path,
                include_subdirectory,
                save_same_dir,
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
        )
        return result_text, result_files

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
        self.task_status_store.update_task(task_id, status="in_progress")

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
        self.task_status_store.update_task(task_id, status="in_progress")

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
                progress=normalized_progress,
                message=message,
                current_item=current_item,
            )
            state["last_progress"] = normalized_progress
            state["last_message"] = message
            state["last_item"] = current_item
            state["last_write_at"] = now

        return callback

    def render_task_monitor_html(self) -> str:
        tasks = self.task_status_store.list_tasks(limit=10)
        active_statuses = {"queued", "in_progress"}
        active_tasks = [task for task in tasks if task["status"] in active_statuses]
        recent_tasks = [task for task in tasks if task["status"] not in active_statuses]

        sections = [
            '<div class="task-monitor">',
            self.render_task_group(
                title="Active Tasks",
                tasks=active_tasks,
                empty_message="No active transcription tasks. This panel refreshes automatically.",
            ),
        ]

        if recent_tasks:
            sections.append(
                self.render_task_group(
                    title="Recent Tasks",
                    tasks=recent_tasks,
                    empty_message="",
                )
            )

        sections.append("</div>")
        return "".join(sections)

    def render_task_group(self, title: str, tasks: list[dict], empty_message: str) -> str:
        if not tasks:
            return (
                '<section class="task-monitor__section">'
                f'<div class="task-monitor__title">{escape(title)}</div>'
                f'<div class="task-monitor__empty">{escape(empty_message)}</div>'
                "</section>"
            )

        cards = "".join(self.render_task_card(task) for task in tasks)
        return (
            '<section class="task-monitor__section">'
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
                    with gr.Row():
                        tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                        files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3, interactive=False)
                        btn_openfolder = gr.Button('📂', scale=1)

                    input_file = gr.Files(type="filepath", label=_("Upload File here"), file_types=MEDIA_EXTENSION)

                    with gr.Row():
                        btn_run = gr.Button("Транскрибация", variant="primary")

                    tb_input_folder = gr.Textbox(label="Input Folder Path (Optional)",
                                                 info="Optional: Specify the folder path where the input files are located, if you prefer to use local files instead of uploading them."
                                                      " Leave this field empty if you do not wish to use a local path.",
                                                 visible=self.args.colab,
                                                 value="")
                    cb_include_subdirectory = gr.Checkbox(label="Include Subdirectory Files",
                                                          info="When using Input Folder Path above, whether to include all files in the subdirectory or not.",
                                                          visible=self.args.colab,
                                                          value=False)
                    cb_save_same_dir = gr.Checkbox(label="Save outputs at same directory",
                                                   info="When using Input Folder Path above, whether to save output in the same directory as inputs or not, in addition to the original"
                                                        " output directory.",
                                                   visible=self.args.colab,
                                                   value=True)

                    pipeline_params, dd_file_format, cb_timestamp = self.create_pipeline_inputs()

                    params = [input_file, tb_input_folder, cb_include_subdirectory, cb_save_same_dir,
                              dd_file_format, cb_timestamp]
                    params = params + pipeline_params
                    btn_run.click(fn=self.transcribe_file_with_task_tracking,
                                  inputs=params,
                                  outputs=[tb_indicator, files_subtitles])
                    btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)

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
            allowed_paths=eval(args.allowed_paths) if args.allowed_paths else None
        )

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
_args = parser.parse_args()

if __name__ == "__main__":
    app = App(args=_args)
    app.launch()
