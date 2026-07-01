from pathlib import Path
from types import SimpleNamespace

import app as app_module
from app import App
from modules.elevenlabs_transcription.models import ElevenLabsSettings, TranscriptionProvider
from modules.utils.files_manager import save_yaml


def bare_app(tmp_path: Path) -> App:
    instance = App.__new__(App)
    instance.args = SimpleNamespace(output_dir=str(tmp_path))
    instance.default_params = {}
    return instance


def test_provider_defaults_to_elevenlabs_and_is_persisted(monkeypatch, tmp_path: Path):
    config = tmp_path / "defaults.yaml"
    save_yaml({"whisper": {}}, str(config))
    monkeypatch.setattr(app_module, "DEFAULT_PARAMETERS_CONFIG_PATH", str(config))
    instance = bare_app(tmp_path)

    assert instance.get_saved_transcription_provider() is TranscriptionProvider.ELEVENLABS
    assert instance.get_saved_elevenlabs_settings().diarize is True

    instance.cache_provider_settings("whisper", ElevenLabsSettings(language_code="ru"))

    assert instance.get_saved_transcription_provider() is TranscriptionProvider.WHISPER
    assert instance.get_saved_elevenlabs_settings().language_code == "ru"
    assert "key" not in instance.get_elevenlabs_config_defaults()


def test_explicit_whisper_dispatch_uses_existing_pipeline(tmp_path: Path):
    instance = bare_app(tmp_path)
    calls = []

    class FakeWhisper:
        def transcribe_file(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "whisper", [str(tmp_path / "result.txt")]

    instance.whisper_inf = FakeWhisper()

    text, paths, actual_provider = instance.transcribe_files_by_provider(
        files=[str(tmp_path / "source.wav")],
        provider="whisper",
        file_format="txt",
        add_timestamp=False,
        output_dir=str(tmp_path),
        progress=None,
        pipeline_params=[],
        elevenlabs_settings=ElevenLabsSettings(),
    )

    assert text == "whisper"
    assert paths == [str(tmp_path / "result.txt")]
    assert actual_provider == "whisper"
    assert len(calls) == 1


def test_folder_input_is_rejected_before_elevenlabs_dispatch(tmp_path: Path):
    instance = bare_app(tmp_path)
    instance.elevenlabs_pipeline = SimpleNamespace(
        transcribe_files=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run"))
    )

    try:
        instance.transcribe_files_by_provider(
            files=[],
            provider="elevenlabs",
            file_format="txt",
            add_timestamp=False,
            output_dir=str(tmp_path),
            progress=None,
            pipeline_params=[],
            elevenlabs_settings=ElevenLabsSettings(),
            input_folder_path=str(tmp_path),
        )
    except ValueError as error:
        assert "only with the Whisper provider" in str(error)
    else:
        raise AssertionError("Folder input should be rejected for ElevenLabs")
