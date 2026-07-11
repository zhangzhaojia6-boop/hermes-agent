from __future__ import annotations

from pathlib import Path

import pytest

from gateway.config import GatewayConfig


def test_sync_xintai_runtime_soul_installs_canonical_content(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

    from gateway.xintai_soul import sync_xintai_runtime_soul

    runtime_home = tmp_path / "hermes-home"
    runtime_home.mkdir(parents=True)

    result = sync_xintai_runtime_soul()

    soul_text = (runtime_home / "SOUL.md").read_text(encoding="utf-8")

    assert result.installed is True
    assert "鑫泰铝业智能大脑" in soul_text
    assert "只用中文" in soul_text
    assert "不要猜任何生产数字" in soul_text
    assert result.reason == "installed_missing"
    assert result.backup_path is None


def test_sync_xintai_runtime_soul_preserves_existing_custom_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

    from gateway.xintai_soul import sync_xintai_runtime_soul

    runtime_home = tmp_path / "hermes-home"
    runtime_home.mkdir(parents=True)
    target = runtime_home / "SOUL.md"
    custom_text = "custom soul line 1\r\ncustom soul line 2\r\n"
    target.write_bytes(custom_text.encode("utf-8"))

    result_first = sync_xintai_runtime_soul()
    result_second = sync_xintai_runtime_soul()

    assert result_first.installed is False
    assert result_first.reason == "existing_preserved"
    assert result_second.installed is False
    assert result_second.reason == "existing_preserved"
    assert target.read_bytes() == custom_text.encode("utf-8")
    assert not any(runtime_home.glob("SOUL.md.xintai-*.bak"))


def test_sync_xintai_runtime_soul_can_be_disabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("XINTAI_SOUL_SYNC_ENABLED", "false")

    from gateway.xintai_soul import sync_xintai_runtime_soul

    runtime_home = tmp_path / "hermes-home"
    runtime_home.mkdir(parents=True)
    target = runtime_home / "SOUL.md"
    target.write_text("keep current soul", encoding="utf-8")

    result = sync_xintai_runtime_soul()

    assert result.installed is False
    assert result.reason == "disabled"
    assert target.read_text(encoding="utf-8") == "keep current soul"


@pytest.mark.asyncio
async def test_start_gateway_syncs_runtime_soul_before_runner(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

    calls: list[str] = []

    class _CleanExitRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.adapters = {}

        async def start(self):
            return True

        async def stop(self):
            return None

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("gateway.run.sync_xintai_runtime_soul", lambda: calls.append("sync"))
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _CleanExitRunner)

    from gateway.run import start_gateway

    ok = await start_gateway(config=GatewayConfig(), replace=False, verbosity=None)

    assert ok is True
    assert calls == ["sync"]
