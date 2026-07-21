from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner, _home_channel_missing_notice
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="channel",
        user_id="U123",
        thread_id="111.222",
    )


def _make_runner(extra=None):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(enabled=True, token="***", extra=extra or {})
        }
    )
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="public-1"))
    adapter.send_private_notice = AsyncMock(return_value=SendResult(success=True, message_id="private-1"))
    runner.adapters = {Platform.SLACK: adapter}
    return runner, adapter


def test_home_channel_notice_is_chinese_when_runtime_language_is_chinese(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")

    notice = _home_channel_missing_notice("dingtalk", "/sethome")

    assert "默认回传频道" in notice
    assert "/sethome" in notice
    assert "No home channel" not in notice


def test_home_channel_notice_preserves_english_default(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")

    notice = _home_channel_missing_notice("dingtalk", "/sethome")

    assert "No home channel is set for Dingtalk" in notice


@pytest.mark.asyncio
async def test_deliver_platform_notice_uses_private_delivery_when_configured():
    runner, adapter = _make_runner(extra={"notice_delivery": "private"})

    await runner._deliver_platform_notice(_make_source(), "hello")

    adapter.send_private_notice.assert_awaited_once_with(
        "C123",
        "U123",
        "hello",
        metadata={"thread_id": "111.222"},
    )
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_deliver_platform_notice_falls_back_to_public_when_private_fails():
    runner, adapter = _make_runner(extra={"notice_delivery": "private"})
    adapter.send_private_notice = AsyncMock(return_value=SendResult(success=False, error="nope"))

    await runner._deliver_platform_notice(_make_source(), "hello")

    adapter.send.assert_awaited_once_with("C123", "hello", metadata={"thread_id": "111.222"})


@pytest.mark.asyncio
async def test_deliver_platform_notice_uses_public_delivery_by_default():
    runner, adapter = _make_runner()

    await runner._deliver_platform_notice(_make_source(), "hello")

    adapter.send.assert_awaited_once_with("C123", "hello", metadata={"thread_id": "111.222"})
    adapter.send_private_notice.assert_not_awaited()
