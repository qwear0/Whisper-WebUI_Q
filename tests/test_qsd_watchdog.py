from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from modules.qsd_api.watchdog import Watchdog, UncertainLaunch


class Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_disabled_without_key(monkeypatch):
    monkeypatch.delenv("QSD_BACKEND_API_KEY", raising=False)
    client = Mock()
    Watchdog(Mock(), client=client).poll()
    client.get.assert_not_called()


def test_active_local_task_blocks_launch(monkeypatch):
    monkeypatch.setenv("QSD_BACKEND_API_KEY", "test-key")
    monkeypatch.setenv("QSD_TRANSCRIPTION_WATCHDOG_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "elevenlabs")
    client = Mock()
    client.get.return_value = Response({"candidates": []})
    store = Mock()
    store.list_active_tasks.return_value = [{"id": "busy"}]
    Watchdog(store, client=client).poll()
    client.post.assert_not_called()


def test_ambiguous_launch_reconciles_on_restart(monkeypatch):
    monkeypatch.setenv("QSD_BACKEND_API_KEY", "test-key")
    monkeypatch.setenv("QSD_TRANSCRIPTION_WATCHDOG_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "elevenlabs")
    client = Mock()
    client.get.return_value = Response({"candidates": [{"request_id": "r", "status": "WAITING_CREDITS", "attempt_generation": 2, "due_at": "2024-01-01T00:00:00Z"}]})
    client.post.side_effect = [Response({"data": {"runsOrError": {"__typename": "Runs", "results": []}}}), Response({"data": {"runsOrError": {"__typename": "Runs", "results": []}}}), TimeoutError()]
    store = Mock()
    store.list_active_tasks.return_value = []
    watcher = Watchdog(store, client=client, clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(UncertainLaunch):
        watcher.poll()
    client.post.side_effect = [Response({"data": {"runsOrError": {"__typename": "Runs", "results": []}}}), Response({"data": {"runsOrError": {"__typename": "Runs", "results": [{"runId": "already-launched", "status": "SUCCESS"}]}}})]
    Watchdog(store, client=client, clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)).poll()
    assert sum("mutation" in call.kwargs["json"]["query"] for call in client.post.call_args_list) == 1
    queries = [call.kwargs["json"] for call in client.post.call_args_list]
    assert queries[0]["variables"]["filter"] == {
        "pipelineName": "telegram_audio_transcription_job",
        "statuses": ["CANCELING", "MANAGED", "NOT_STARTED", "QUEUED", "STARTED", "STARTING"],
    }
    assert queries[1]["variables"]["filter"] == {
        "pipelineName": "telegram_audio_transcription_job",
        "tags": [
            {"key": "request_id", "value": "r"},
            {"key": "expected_attempt_generation", "value": "2"},
        ],
    }
    launch = next(query for query in queries if "mutation" in query["query"])
    assert launch["variables"]["params"]["executionMetadata"]["tags"][-1] == {
        "key": "qsd/capacity_lane", "value": "general",
    }


@pytest.mark.parametrize("terminal_run_count", [99, 100])
def test_terminal_run_history_does_not_block_dispatch(monkeypatch, terminal_run_count):
    monkeypatch.setenv("QSD_BACKEND_API_KEY", "test-key")
    monkeypatch.setenv("QSD_TRANSCRIPTION_WATCHDOG_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "elevenlabs")
    candidate = {
        "request_id": "eligible-request",
        "status": "QUEUED",
        "attempt_generation": 1,
        "due_at": None,
    }
    history = [
        {"runId": f"terminal-{index}", "status": "SUCCESS"}
        for index in range(terminal_run_count)
    ]
    client = Mock()
    client.get.return_value = Response({"candidates": [candidate]})

    def graphql_response(_url, *, json):
        query = json["query"]
        if "mutation" in query:
            return Response({"data": {"launchPipelineExecution": {
                "__typename": "LaunchRunSuccess", "run": {"runId": "launched"},
            }}})
        filters = json["variables"]["filter"]
        if "tags" in filters:
            return Response({"data": {"runsOrError": {"__typename": "Runs", "results": []}}})
        assert filters == {
            "pipelineName": "telegram_audio_transcription_job",
            "statuses": ["CANCELING", "MANAGED", "NOT_STARTED", "QUEUED", "STARTED", "STARTING"],
        }
        active_statuses = set(filters["statuses"])
        active_results = [run for run in history if run["status"] in active_statuses]
        return Response({"data": {"runsOrError": {
            "__typename": "Runs", "results": active_results[:100],
        }}})

    client.post.side_effect = graphql_response
    store = Mock()
    store.list_active_tasks.return_value = []

    Watchdog(store, client=client, clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)).poll()

    assert sum("mutation" in call.kwargs["json"]["query"] for call in client.post.call_args_list) == 1
    assert len(client.post.call_args_list) == 3


@pytest.mark.parametrize("active_run_count", [1, 100])
def test_active_dagster_runs_fail_closed(monkeypatch, active_run_count):
    monkeypatch.setenv("QSD_BACKEND_API_KEY", "test-key")
    monkeypatch.setenv("QSD_TRANSCRIPTION_WATCHDOG_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "elevenlabs")
    client = Mock()
    client.get.return_value = Response({"candidates": [{
        "request_id": "eligible-request", "status": "QUEUED", "attempt_generation": 1, "due_at": None,
    }]})
    active_statuses = ["QUEUED", "NOT_STARTED", "MANAGED", "STARTING", "STARTED", "CANCELING"]
    client.post.return_value = Response({"data": {"runsOrError": {
        "__typename": "Runs",
        "results": [
            {"runId": f"active-{index}", "status": active_statuses[index % len(active_statuses)]}
            for index in range(active_run_count)
        ],
    }}})
    store = Mock()
    store.list_active_tasks.return_value = []

    Watchdog(store, client=client).poll()

    client.post.assert_called_once()
    assert "mutation" not in client.post.call_args.kwargs["json"]["query"]


def test_candidate_backend_order_is_preserved_after_due_filtering(monkeypatch):
    monkeypatch.setenv("QSD_BACKEND_API_KEY", "test-key")
    monkeypatch.setenv("QSD_TRANSCRIPTION_WATCHDOG_ENABLED", "true")
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "elevenlabs")
    client = Mock()
    client.get.return_value = Response({"candidates": [
        {
            "request_id": "not-yet-due",
            "status": "WAITING_CREDITS",
            "attempt_generation": 1,
            "due_at": "2030-01-01T00:00:00Z",
        },
        {
            "request_id": "older-due-large-audio",
            "status": "WAITING_CREDITS",
            "attempt_generation": 2,
            "due_at": "2024-01-01T00:00:00Z",
            "audio_size_bytes": 8_000_000,
        },
        {
            "request_id": "newer-queued-small-audio",
            "status": "QUEUED",
            "attempt_generation": 1,
            "due_at": None,
            "audio_size_bytes": 24_000,
        },
    ]})
    client.post.side_effect = [
        Response({"data": {"runsOrError": {"__typename": "Runs", "results": []}}}),
        Response({"data": {"runsOrError": {"__typename": "Runs", "results": []}}}),
        Response({"data": {"launchPipelineExecution": {
            "__typename": "LaunchRunSuccess", "run": {"runId": "launched"},
        }}}),
    ]
    store = Mock()
    store.list_active_tasks.return_value = []

    Watchdog(store, client=client, clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)).poll()

    launch = next(call.kwargs["json"] for call in client.post.call_args_list if "mutation" in call.kwargs["json"]["query"])
    config = launch["variables"]["params"]["runConfigData"]["ops"]["run_telegram_audio_transcription"]["config"]
    assert config == {"request_id": "older-due-large-audio", "attempt_generation": 2}


def test_disabled_for_wrong_provider_or_missing_explicit_gate(monkeypatch):
    monkeypatch.setenv("QSD_BACKEND_API_KEY", "test-key")
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "whisper")
    monkeypatch.setenv("QSD_TRANSCRIPTION_WATCHDOG_ENABLED", "true")
    client = Mock()
    Watchdog(Mock(), client=client).poll()
    client.get.assert_not_called()
    monkeypatch.setenv("TRANSCRIBE_PROVIDER", "elevenlabs")
    monkeypatch.delenv("QSD_TRANSCRIPTION_WATCHDOG_ENABLED")
    Watchdog(Mock(), client=client).poll()
    client.get.assert_not_called()
