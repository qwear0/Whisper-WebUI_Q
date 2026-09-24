"""Opt-in, fail-closed QSD dispatch poller. No audio or Telegram credentials cross this boundary."""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone

import httpx


BACKEND = "http://qsd_backend:8000/transcription/dispatch-candidates"
DAGSTER = "http://enigma_dagster_webserver:3000/graphql"
RUNS = """query($filter: RunsFilter!) { runsOrError(filter: $filter, limit: 100) {
  __typename ... on Runs { results { runId status } }
} }"""
LAUNCH = """mutation($params: ExecutionParams!) { launchPipelineExecution(executionParams: $params) {
  __typename ... on LaunchRunSuccess { run { runId } }
} }"""
ACTIVE = {"QUEUED", "NOT_STARTED", "MANAGED", "STARTING", "STARTED", "CANCELING"}
TERMINAL = {"SUCCESS", "FAILURE", "CANCELED"}


class UncertainLaunch(Exception):
    pass


class Watchdog:
    def __init__(self, store, *, client=None, clock=None, interval=30):
        self.store = store
        self.client = client or httpx.Client(timeout=10)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.thread is not None:
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, name="qsd-dispatch-watchdog", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=15)
            self.thread = None
        self.client.close()

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self.poll()
            except Exception:
                # The next poll is safe only for read failures; ambiguous launches remain held.
                pass
            self.stop_event.wait(self.interval)

    def _graphql(self, query, variables):
        response = self.client.post(DAGSTER, json={"query": query, "variables": variables})
        response.raise_for_status()
        data = response.json()
        if data.get("errors") or not isinstance(data.get("data"), dict):
            raise ValueError("Dagster query failed")
        return data["data"]

    def _runs(self, tags=None):
        filter_value = {"pipelineName": "telegram_audio_transcription_job"}
        if tags is None:
            # The global capacity check is bounded to active runs; tag reconciliation includes terminal history.
            filter_value["statuses"] = sorted(ACTIVE)
        else:
            filter_value["tags"] = tags
        value = self._graphql(RUNS, {"filter": filter_value})["runsOrError"]
        if value.get("__typename") != "Runs" or not isinstance(value.get("results"), list):
            raise ValueError("Dagster runs unavailable")
        return value["results"]

    def poll(self):
        key = os.environ.get("QSD_BACKEND_API_KEY")
        if (not key or os.environ.get("QSD_TRANSCRIPTION_WATCHDOG_ENABLED") != "true"
                or os.environ.get("TRANSCRIBE_PROVIDER") != "elevenlabs"):
            return
        response = self.client.get(BACKEND, params={"limit": 20}, headers={"X-QSD-API-Key": key})
        response.raise_for_status()
        candidates = response.json()["candidates"]
        if not isinstance(candidates, list) or len(candidates) > 20:
            raise ValueError("Invalid candidate batch")
        if self.store.list_active_tasks(limit=1):
            return
        active_runs = self._runs()
        if len(active_runs) >= 100 or any(run.get("status") not in TERMINAL for run in active_runs):
            return
        now = self.clock()
        eligible = []
        for row in candidates:
            if row["status"] not in ("QUEUED", "WAITING_CREDITS") or not isinstance(row["attempt_generation"], int):
                raise ValueError("Invalid candidate")
            due = row.get("due_at")
            if due is not None:
                due_date = datetime.fromisoformat(due.replace("Z", "+00:00"))
                if due_date > now:
                    continue
            eligible.append((row["request_id"], row))
        for request_id, row in eligible:
            generation = row["attempt_generation"]
            tags = [{"key": "request_id", "value": request_id}, {"key": "expected_attempt_generation", "value": str(generation)}]
            matches = self._runs(tags)
            if matches:
                continue
            if self.store.list_active_tasks(limit=1):
                return
            params = {"selector": {"repositoryLocationName": os.environ.get("DAGSTER_REPO_LOCATION_NAME", "enigma"), "repositoryName": os.environ.get("DAGSTER_REPO_NAME", "__repository__"), "pipelineName": "telegram_audio_transcription_job"}, "runConfigData": {"ops": {"run_telegram_audio_transcription": {"config": {"request_id": request_id, "attempt_generation": generation}}}}, "mode": "default", "executionMetadata": {"tags": tags + [{"key": "qsd/capacity_lane", "value": "general"}]}}
            # Always reconcile by tags before launch, including after a restart or timeout.
            try:
                result = self._graphql(LAUNCH, {"params": params})["launchPipelineExecution"]
                if result.get("__typename") != "LaunchRunSuccess" or not result.get("run", {}).get("runId"):
                    raise UncertainLaunch("Launch outcome requires manual triage")
            except Exception as error:
                raise UncertainLaunch("Launch outcome requires reconciliation before retry") from error
            return
