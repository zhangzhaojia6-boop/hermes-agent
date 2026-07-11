from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.xintai_evidence_relay import (
    RelayResult,
    XintaiEvidenceRelay,
    build_dingtalk_fallback_identity,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


@pytest.mark.asyncio
async def test_relay_posts_follow_up_phrase_without_legacy_keyword() -> None:
    client = AsyncMock()
    client.post = AsyncMock(
        return_value=_FakeResponse(200, {"errcode": 0, "trace_id": "trace-follow-up-001"})
    )
    relay = XintaiEvidenceRelay(
        enabled=True,
        base_url="https://datahub.example",
        inbound_token="relay-token",
        http_client=client,
        retry_backoff_seconds=(0, 0, 0),
    )

    result = await relay.relay(
        {
            "conversationId": "cid-follow-up",
            "conversationType": "group",
            "senderStaffId": "staff-001",
            "messageId": "msg-follow-up-001",
            "traceId": "trace-follow-up-001",
            "msgtype": "text",
            "text": {"content": "昨天那个先继续跟一下"},
        }
    )

    assert result == RelayResult(accepted=True, trace_id="trace-follow-up-001", category="accepted", error=None)
    call = client.post.call_args
    assert call.args[0] == "https://datahub.example/api/v1/dingtalk/agent-inbound"
    assert call.kwargs["headers"] == {"x-dingtalk-inbound-token": "relay-token"}
    assert call.kwargs["json"]["text"]["content"] == "昨天那个先继续跟一下"
    assert call.kwargs["json"]["traceId"] == "trace-follow-up-001"
    assert call.kwargs["timeout"] == 8.0


@pytest.mark.asyncio
async def test_relay_returns_explicit_disabled_result() -> None:
    relay = XintaiEvidenceRelay(enabled=False, base_url="https://datahub.example", inbound_token="relay-token")

    result = await relay.relay({"messageId": "msg-disabled-001"})

    assert result.accepted is False
    assert result.trace_id == "msg-disabled-001"
    assert result.category == "disabled"


@pytest.mark.asyncio
async def test_relay_returns_explicit_misconfigured_result() -> None:
    relay = XintaiEvidenceRelay(enabled=True, base_url="", inbound_token="")

    result = await relay.relay({"messageId": "msg-misconfigured-001"})

    assert result.accepted is False
    assert result.trace_id == "msg-misconfigured-001"
    assert result.category == "misconfigured"


@pytest.mark.asyncio
async def test_relay_retries_then_succeeds(monkeypatch) -> None:
    client = AsyncMock()
    client.post = AsyncMock(
        side_effect=[
            httpx.ConnectError("first failure"),
            httpx.ReadTimeout("second failure"),
            _FakeResponse(200, {"errcode": 0, "trace_id": "trace-retry-001"}),
        ]
    )
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("gateway.xintai_evidence_relay.asyncio.sleep", _fake_sleep)
    relay = XintaiEvidenceRelay(
        enabled=True,
        base_url="https://datahub.example",
        inbound_token="relay-token",
        http_client=client,
        retry_backoff_seconds=(0.01, 0.02, 0.03),
    )

    result = await relay.relay({"traceId": "trace-retry-001", "messageId": "msg-retry-001"})

    assert result.accepted is True
    assert result.category == "accepted"
    assert client.post.await_count == 3
    assert sleep_calls == [0.01, 0.02]


@pytest.mark.asyncio
async def test_relay_failure_redacts_token_from_error_and_logs(caplog) -> None:
    secret = "relay-token-secret"
    client = AsyncMock()
    client.post = AsyncMock(
        return_value=_FakeResponse(
            500,
            payload={"errcode": 500, "errmsg": f"Authorization: Bearer {secret}"},
            text=f"Authorization: Bearer {secret}; downloadCode=download-secret-001",
        )
    )
    relay = XintaiEvidenceRelay(
        enabled=True,
        base_url="https://datahub.example",
        inbound_token=secret,
        http_client=client,
        retry_backoff_seconds=(0, 0, 0),
    )

    with caplog.at_level("WARNING"):
        result = await relay.relay({"traceId": "trace-redact-001", "messageId": "msg-redact-001"})

    assert result.accepted is False
    assert result.category == "http_500"
    assert secret not in (result.error or "")
    assert secret not in caplog.text
    assert "download-secret-001" not in (result.error or "")
    assert "download-secret-001" not in caplog.text


def test_build_dingtalk_fallback_identity_is_stable_for_text_events() -> None:
    payload = {
        "conversationId": "cid-text-001",
        "conversationType": "2",
        "senderId": "sender-001",
        "senderStaffId": "staff-001",
        "createAt": "1720688400000",
        "msgtype": "text",
        "text": {"content": "昨天那个先继续跟一下"},
        "messageId": None,
    }

    repeated = dict(payload)
    changed = dict(payload, text={"content": "今天这个改成新的"})

    assert build_dingtalk_fallback_identity(payload) == build_dingtalk_fallback_identity(repeated)
    assert build_dingtalk_fallback_identity(payload) != build_dingtalk_fallback_identity(changed)


def test_build_dingtalk_fallback_identity_ignores_download_secrets() -> None:
    payload = {
        "conversationId": "cid-file-001",
        "conversationType": "2",
        "senderId": "sender-001",
        "senderStaffId": "staff-001",
        "createAt": "1720688400000",
        "msgtype": "file",
        "fileName": "日报.xlsx",
        "fileId": "file-001",
        "downloadCode": "download-secret-001",
        "sessionWebhook": "https://api.dingtalk.com/webhook?access_token=secret-001",
        "messageId": None,
    }

    repeated_with_new_secret = dict(
        payload,
        downloadCode="download-secret-002",
        sessionWebhook="https://api.dingtalk.com/webhook?access_token=secret-002",
    )
    changed_file = dict(payload, fileId="file-002")

    first = build_dingtalk_fallback_identity(payload)
    repeated = build_dingtalk_fallback_identity(repeated_with_new_secret)
    changed = build_dingtalk_fallback_identity(changed_file)

    assert first == repeated
    assert first != changed
    assert "download-secret" not in first
    assert "secret-001" not in first
    assert first.startswith("dingtalk-stream-sha256:")
