"""Cold deployment fixtures: no Docker, models, jobs, credentials or live HTTP."""
import fcntl
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ID = "sha256:" + "a" * 64
ARGS = ("--image=whisper-webui=" + IMAGE_ID, "--proxy-ip=172.27.0.1")


@pytest.fixture
def deployment(tmp_path):
    script = tmp_path / "apply_prebuilt.sh"
    script.write_bytes((ROOT / "apply_prebuilt.sh").read_bytes())
    log = tmp_path / "commands"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scripts = {
        "git": '''#!/bin/bash
printf 'git %s\\n' "$*" >> "$COMMAND_LOG"
case "$*" in
 *symbolic-ref*) echo "${BRANCH:-master}";;
 *status*)
  n=0; [[ ! -f "$COMMAND_LOG.status" ]] || n=$(cat "$COMMAND_LOG.status")
  n=$((n+1)); echo "$n" > "$COMMAND_LOG.status"
  if [[ "$LATE_DRIFT" == 1 && "$n" == 2 ]]; then touch "$COMMAND_LOG.late"; fi
  [[ "$STATUS_FAIL" != 1 ]] || exit 9
  [[ "$DIRTY" != 1 ]] || echo ' M foreign';;
 *rev-parse*)
  if [[ -f "$COMMAND_LOG.checked" && "$HEAD_DRIFT" == 1 ]]; then echo changed; else echo sealed; fi;;
esac
''',
        "docker": '''#!/bin/bash
printf 'docker %s\\n' "$*" >> "$COMMAND_LOG"
[[ -z "$COMPOSE_PROFILES" ]] || exit 19
case "$*" in
 info) exit "${DAEMON_FAIL:-0}";;
 *'config --quiet'*) exit "${CONFIG_FAIL:-0}";;
 *'config --format json'*)
  if [[ -f "$COMMAND_LOG.checked" && "$CONFIG_DRIFT" == 1 ]]; then echo changed
  else echo "$CONFIG_JSON"; fi;;
 'image inspect'*)
  n=0; [[ ! -f "$COMMAND_LOG.images" ]] || n=$(cat "$COMMAND_LOG.images")
  n=$((n+1)); echo "$n" > "$COMMAND_LOG.images"; touch "$COMMAND_LOG.checked"
  [[ "$MISSING" != 1 ]] || exit 12
  if [[ "$WRONG" == 1 || ("$DRIFT" == 1 && "$n" == 2) || -f "$COMMAND_LOG.late" ]]; then echo 'sha256:wrong'; else echo "$IMAGE_ID"; fi;;
 *' up '*) exit "${UP_FAIL:-0}";;
esac
''',
    }
    for name, source in scripts.items():
        p = bin_dir / name
        p.write_text(source)
        p.chmod(0o755)

    def run(*arguments, **overrides):
        for p in tmp_path.glob("commands*"):
            p.unlink()
        config = {"services": {"whisper-webui": {"image": IMAGE_ID, "environment": {
            "FORWARDED_ALLOW_IPS": "127.0.0.1,172.27.0.1", "PRIVATE_FIXTURE": "MUST-NOT-EMIT"}}}}
        env = os.environ | {"PATH": f"{bin_dir}:{os.environ['PATH']}", "COMMAND_LOG": str(log),
                            "CONFIG_JSON": json.dumps(config), "IMAGE_ID": IMAGE_ID} | overrides
        result = subprocess.run(["bash", str(script), *arguments], env=env, cwd="/",
                                text=True, capture_output=True, timeout=15)
        calls = log.read_text() if log.exists() else ""
        assert "MUST-NOT-EMIT" not in result.stdout + result.stderr + calls
        return result, calls

    return run, tmp_path


def test_prebuilt_uses_only_existing_image_and_preserves_role_scope(deployment):
    run, _ = deployment
    result, calls = run(*ARGS, COMPOSE_PROFILES="other")
    assert result.returncode == 0, result.stderr
    assert calls.count("image inspect") == 2
    assert "up -d --no-build --pull never --wait --wait-timeout 300 --timeout 120 whisper-webui" in calls
    assert calls.rindex("image inspect") > calls.rindex("status --porcelain")
    assert " build " not in calls and " push " not in calls and " run " not in calls
    assert "--force-recreate" not in calls and "--remove-orphans" not in calls


@pytest.mark.parametrize("setting", [
    {"WRONG": "1"}, {"MISSING": "1"}, {"DRIFT": "1"}, {"DIRTY": "1"},
    {"STATUS_FAIL": "1"}, {"BRANCH": "main"}, {"HEAD_DRIFT": "1"},
    {"CONFIG_DRIFT": "1"}, {"DAEMON_FAIL": "1"}, {"CONFIG_FAIL": "1"},
    {"LATE_DRIFT": "1"},
])
def test_invalid_inputs_never_apply(deployment, setting):
    run, root = deployment
    result, calls = run(*ARGS, **setting)
    assert result.returncode != 0
    assert " up " not in calls
    if setting.get("LATE_DRIFT"):
        assert (root / "commands.late").exists()
        assert "identity mismatch" in result.stderr
        assert calls.rindex("image inspect") > calls.rindex("status --porcelain")


@pytest.mark.parametrize("ip", ["*", "172.27.0.0/16", "172.27.0.1,172.22.0.1", "127.0.0.1",
                               "0.0.0.0", "8.8.8.8", "192.0.2.1", "::1", "PRIVATE-FIXTURE"])
def test_rejects_broad_public_or_malformed_proxy_without_private_argument_echo(deployment, ip):
    run, _ = deployment
    result, calls = run(ARGS[0], "--proxy-ip=" + ip)
    assert result.returncode == 2
    assert "docker " not in calls
    assert "PRIVATE-FIXTURE" not in result.stderr


@pytest.mark.parametrize("args", [(), (ARGS[0],), (ARGS[1],), (*ARGS, ARGS[0]),
                                  (*ARGS, ARGS[1]), (*ARGS, "--build-only"),
                                  (*ARGS, "--push"), (*ARGS, "--force-recreate"),
                                  ("--image=other=" + IMAGE_ID, ARGS[1])])
def test_rejects_incomplete_or_conflicting_scope(deployment, args):
    run, _ = deployment
    result, calls = run(*args)
    assert result.returncode == 2 and calls == ""


@pytest.mark.parametrize("trust", ["*", "172.27.0.1", "127.0.0.1,172.22.0.1", "127.0.0.1"])
def test_resolved_proxy_trust_must_match_selected_peer(deployment, trust):
    run, _ = deployment
    config = {"services": {"whisper-webui": {"image": IMAGE_ID, "environment": {
        "FORWARDED_ALLOW_IPS": trust}}}}
    result, calls = run(*ARGS, CONFIG_JSON=json.dumps(config))
    assert result.returncode != 0 and " up " not in calls
    assert "sealed private hop" in result.stderr


def test_apply_requires_immutable_image_override(deployment):
    run, _ = deployment
    config = {"services": {"whisper-webui": {"image": "mutable:tag", "environment": {
        "FORWARDED_ALLOW_IPS": "127.0.0.1,172.27.0.1"}}}}
    result, calls = run(*ARGS, CONFIG_JSON=json.dumps(config))
    assert result.returncode != 0 and " up " not in calls
    assert "sealed immutable ID" in result.stderr


def test_lock_blocks_and_up_failure_has_no_automatic_recovery(deployment):
    run, root = deployment
    with (root / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, calls = run(*ARGS)
        assert result.returncode != 0 and calls == ""
    result, calls = run(*ARGS, UP_FAIL="1")
    assert result.returncode != 0
    assert calls.count(" up ") == 1 and " down " not in calls


def test_compose_retains_existing_auth_data_networks_and_launch():
    c = yaml.safe_load((ROOT / "docker-compose.yaml").read_text())
    s = c["services"]["whisper-webui"]
    assert s["environment"]["QSD_WHISPER_API_KEY"] == "${QSD_WHISPER_API_KEY:-}"
    assert s["environment"]["FORWARDED_ALLOW_IPS"] == "127.0.0.1,${WHISPER_GATEWAY_PROXY_IP:-127.0.0.1}"
    assert s["networks"] == ["default", "enigma_control_plane"]
    assert s["entrypoint"] == ["python", "app.py", "--server_port", "7860", "--server_name", "0.0.0.0"]
    assert s["image"] == "${WHISPER_PREBUILT_IMAGE:-jhj0517/whisper-webui:latest}"
    assert s["stop_grace_period"] == "2m"
    assert s["volumes"] == ["./models:/Whisper-WebUI/models", "./outputs:/Whisper-WebUI/outputs",
                            "./configs:/Whisper-WebUI/configs",
                            "/data/ObsidianVault/Inner/AI/Whisper_Results:/data/ObsidianVault/Inner/AI/Whisper_Results"]
    assert ".build.lock" in (ROOT / ".gitignore").read_text().splitlines()
