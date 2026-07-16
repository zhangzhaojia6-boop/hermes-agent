from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from plugins.platforms.dingtalk.adapter import DingTalkAdapter
from plugins.platforms.dingtalk import adapter as dingtalk_module
from gateway.session import SessionEntry, SessionSource, build_session_key
from gateway.xintai_callback_proof_ledger import get_proof_ledger_path
from gateway.xintai_evidence_relay import RelayResult
from gateway.xintai_soul import sync_xintai_runtime_soul


def _make_dingtalk_message(*, text: str = "", payload: dict | None = None, **overrides):
    payload = dict(payload or {})
    msg = SimpleNamespace()
    msg.message_id = overrides.get("message_id", "msg-001")
    msg.conversation_id = overrides.get("conversation_id", "cid-001")
    msg.conversation_type = overrides.get("conversation_type", "2")
    msg.sender_id = overrides.get("sender_id", "sender-001")
    msg.sender_staff_id = overrides.get("sender_staff_id", "staff-001")
    msg.sender_union_id = overrides.get("sender_union_id", "union-001")
    msg.sender_nick = overrides.get("sender_nick", "测试员")
    msg.conversation_title = overrides.get("conversation_title", "测试群")
    msg.session_webhook = overrides.get("session_webhook", "https://api.dingtalk.com/webhook")
    msg.create_at = overrides.get("create_at", "1720688400000")
    msg.text = {"content": text} if text else ""
    msg.rich_text = None
    msg.rich_text_content = None
    msg.data = {
        "messageId": msg.message_id,
        "conversationId": msg.conversation_id,
        "conversationType": msg.conversation_type,
        "senderId": msg.sender_id,
        "senderStaffId": msg.sender_staff_id,
        "senderUnionId": msg.sender_union_id,
        "senderNick": msg.sender_nick,
        "conversationTitle": msg.conversation_title,
        "sessionWebhook": msg.session_webhook,
        "createAt": msg.create_at,
        "msgtype": payload.get("msgtype", "text"),
        **payload,
    }
    return msg


def _make_connected_adapter() -> DingTalkAdapter:
    adapter = DingTalkAdapter(PlatformConfig(enabled=True))
    adapter._accepting_events = True
    adapter._shutting_down = False
    adapter._mark_connected()
    return adapter


def _read_callback_proof_entries() -> list[dict[str, str]]:
    payload = json.loads(get_proof_ledger_path().read_text(encoding="utf-8"))
    return payload["entries"]


@pytest.mark.asyncio
async def test_dingtalk_text_event_relays_before_hermes_handler() -> None:
    adapter = _make_connected_adapter()
    adapter._xintai_evidence_relay = SimpleNamespace(
        relay=AsyncMock(return_value=RelayResult(True, "msg-follow-up-001", "accepted"))
    )
    adapter.handle_message = AsyncMock()

    message = _make_dingtalk_message(
        text="昨天那个先继续跟一下",
        payload={"msgtype": "text"},
        message_id="msg-follow-up-001",
    )

    await adapter._on_message(message)
    tasks = list(adapter._xintai_relay_tasks)
    if tasks:
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)

    relay_payload = adapter._xintai_evidence_relay.relay.await_args.args[0]
    assert relay_payload["msgtype"] == "text"
    assert relay_payload["text"]["content"] == "昨天那个先继续跟一下"
    assert relay_payload["traceId"] == "msg-follow-up-001"
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert isinstance(event, MessageEvent)
    assert event.text == "昨天那个先继续跟一下"
    assert adapter._xintai_stream_health.relay_success_count == 1
    assert adapter._xintai_stream_health.last_event_at is not None


@pytest.mark.asyncio
async def test_non_allowlisted_user_event_still_relays_as_evidence() -> None:
    adapter = DingTalkAdapter(
        PlatformConfig(enabled=True, extra={"allowed_users": ["manager-only"]})
    )
    adapter._accepting_events = True
    adapter._mark_connected()
    adapter._xintai_evidence_relay = SimpleNamespace(
        relay=AsyncMock(return_value=RelayResult(True, "msg-unlisted-001", "accepted"))
    )
    adapter.handle_message = AsyncMock()

    message = _make_dingtalk_message(
        text="今天产量是 123 吨",
        payload={"msgtype": "text"},
        message_id="msg-unlisted-001",
        sender_id="ordinary-user",
        sender_staff_id="ordinary-staff",
    )

    await adapter._on_message(message)
    await asyncio.gather(*list(adapter._xintai_relay_tasks))

    adapter._xintai_evidence_relay.relay.assert_awaited_once()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_unmentioned_group_event_still_relays_as_evidence() -> None:
    adapter = DingTalkAdapter(
        PlatformConfig(enabled=True, extra={"require_mention": True})
    )
    adapter._accepting_events = True
    adapter._mark_connected()
    adapter._xintai_evidence_relay = SimpleNamespace(
        relay=AsyncMock(return_value=RelayResult(True, "msg-unmentioned-001", "accepted"))
    )
    adapter.handle_message = AsyncMock()

    message = _make_dingtalk_message(
        text="日报文件已经发群里了",
        payload={"msgtype": "text", "isInAtList": False},
        message_id="msg-unmentioned-001",
    )
    message.is_in_at_list = False

    await adapter._on_message(message)
    await asyncio.gather(*list(adapter._xintai_relay_tasks))

    adapter._xintai_evidence_relay.relay.assert_awaited_once()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_slow_relay_does_not_delay_hermes_handler() -> None:
    adapter = _make_connected_adapter()
    release_relay = asyncio.Event()
    relay_started = asyncio.Event()

    async def _slow_relay(_payload):
        relay_started.set()
        await release_relay.wait()
        return RelayResult(True, "msg-slow-relay-001", "accepted")

    adapter._xintai_evidence_relay = SimpleNamespace(relay=_slow_relay)
    adapter.handle_message = AsyncMock()

    message = _make_dingtalk_message(
        text="继续看昨天那个产量",
        payload={"msgtype": "text"},
        message_id="msg-slow-relay-001",
    )

    await asyncio.wait_for(adapter._on_message(message), timeout=0.1)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert relay_started.is_set() is True
    adapter.handle_message.assert_awaited_once()
    assert adapter._xintai_stream_health.relay_success_count == 0
    assert adapter._xintai_stream_health.relay_failure_count == 0

    release_relay.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert adapter._xintai_stream_health.relay_success_count == 1
    assert len(adapter._xintai_relay_tasks) == 0


@pytest.mark.asyncio
async def test_dingtalk_file_event_relays_instead_of_skipping() -> None:
    adapter = _make_connected_adapter()
    adapter._xintai_evidence_relay = SimpleNamespace(
        relay=AsyncMock(return_value=RelayResult(True, "msg-file-001", "accepted"))
    )
    adapter.handle_message = AsyncMock()

    message = _make_dingtalk_message(
        text="",
        payload={
            "msgtype": "file",
            "fileName": "日报.xlsx",
            "fileId": "file-001",
            "downloadCode": "download-code-001",
        },
        message_id="msg-file-001",
    )

    await adapter._on_message(message)
    tasks = list(adapter._xintai_relay_tasks)
    if tasks:
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)

    relay_payload = adapter._xintai_evidence_relay.relay.await_args.args[0]
    assert relay_payload["msgtype"] == "file"
    assert relay_payload["fileName"] == "日报.xlsx"
    assert relay_payload["fileId"] == "file-001"
    assert relay_payload["downloadCode"] == "download-code-001"
    assert relay_payload["receivedAt"] == "2024-07-11T09:00:00+00:00"
    assert relay_payload["received_at"] == "2024-07-11T09:00:00+00:00"
    assert relay_payload["messageTime"] == "1720688400000"
    assert relay_payload["msgCreateTime"] == "1720688400000"
    assert relay_payload["createTime"] == "1720688400000"
    assert relay_payload["eventTime"] == "1720688400000"
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_dingtalk_unknown_event_still_relays() -> None:
    adapter = _make_connected_adapter()
    adapter._xintai_evidence_relay = SimpleNamespace(
        relay=AsyncMock(return_value=RelayResult(False, "msg-unknown-001", "http_400", "rejected"))
    )
    adapter.handle_message = AsyncMock()

    message = _make_dingtalk_message(
        text="",
        payload={"msgtype": "custom_unknown", "eventType": "robot_notice"},
        message_id="msg-unknown-001",
    )

    await adapter._on_message(message)
    tasks = list(adapter._xintai_relay_tasks)
    if tasks:
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)

    relay_payload = adapter._xintai_evidence_relay.relay.await_args.args[0]
    assert relay_payload["msgtype"] == "custom_unknown"
    assert relay_payload["eventType"] == "robot_notice"
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_relay_failure_does_not_block_hermes_handler() -> None:
    adapter = _make_connected_adapter()
    adapter._xintai_evidence_relay = SimpleNamespace(
        relay=AsyncMock(return_value=RelayResult(False, "msg-failure-001", "http_500", "relay failed"))
    )
    adapter.handle_message = AsyncMock()

    message = _make_dingtalk_message(
        text="产量怎么样",
        payload={"msgtype": "text"},
        message_id="msg-failure-001",
    )

    await adapter._on_message(message)
    tasks = list(adapter._xintai_relay_tasks)
    if tasks:
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)

    adapter.handle_message.assert_awaited_once()
    assert adapter._xintai_stream_health.relay_failure_count == 1
    assert adapter._xintai_stream_health.last_error == "relay failed"


@pytest.mark.asyncio
async def test_relay_task_exception_is_consumed_without_duplicate_reply() -> None:
    adapter = _make_connected_adapter()

    async def _boom(_payload):
        raise RuntimeError("relay exploded")

    adapter._xintai_evidence_relay = SimpleNamespace(relay=_boom)
    adapter.handle_message = AsyncMock()

    message = _make_dingtalk_message(
        text="今天产量呢",
        payload={"msgtype": "text"},
        message_id="msg-relay-exception-001",
    )

    await adapter._on_message(message)
    tasks = list(adapter._xintai_relay_tasks)
    if tasks:
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)

    adapter.handle_message.assert_awaited_once()
    assert adapter._xintai_stream_health.relay_failure_count == 1
    assert "relay exploded" in (adapter._xintai_stream_health.last_error or "")
    assert len(adapter._xintai_relay_tasks) == 0


@pytest.mark.asyncio
async def test_disabled_relay_does_not_count_as_failure_or_warn(caplog) -> None:
    adapter = _make_connected_adapter()
    adapter._xintai_evidence_relay = SimpleNamespace(
        relay=AsyncMock(return_value=RelayResult(False, "msg-disabled-001", "disabled"))
    )
    adapter.handle_message = AsyncMock()

    message = _make_dingtalk_message(
        text="继续看昨天那个",
        payload={"msgtype": "text"},
        message_id="msg-disabled-001",
    )

    with caplog.at_level("WARNING"):
        await adapter._on_message(message)

    adapter.handle_message.assert_awaited_once()
    assert adapter._xintai_stream_health.relay_success_count == 0
    assert adapter._xintai_stream_health.relay_failure_count == 0
    assert adapter._xintai_stream_health.last_error is None
    assert "Xintai relay rejected" not in caplog.text
    assert adapter.get_xintai_stream_health()["connected"] is False


@pytest.mark.asyncio
async def test_disconnect_drops_late_callback_before_threadsafe_dispatch(monkeypatch) -> None:
    adapter = _make_connected_adapter()
    adapter.handle_message = AsyncMock()

    inflight_started = asyncio.Event()
    inflight_cancelled = asyncio.Event()
    relay_calls: list[str] = []

    async def _relay(payload):
        relay_calls.append(payload["traceId"])
        if payload["traceId"] == "msg-inflight-001":
            inflight_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                inflight_cancelled.set()
                raise
        return RelayResult(True, payload["traceId"], "accepted")

    adapter._xintai_evidence_relay = SimpleNamespace(relay=_relay, bind_http_client=lambda _client: None)
    websocket_close = AsyncMock()
    adapter._stream_client = SimpleNamespace(websocket=SimpleNamespace(close=websocket_close))
    adapter._stream_task = asyncio.create_task(asyncio.sleep(60))

    inflight_message = _make_dingtalk_message(
        text="",
        payload={"msgtype": "file", "fileName": "日报.xlsx", "fileId": "file-inflight-001"},
        message_id="msg-inflight-001",
    )
    await adapter._on_message(inflight_message)
    await asyncio.sleep(0)
    await asyncio.wait_for(inflight_started.wait(), timeout=0.2)

    late_message = _make_dingtalk_message(
        text="这条消息不该在关停时继续处理",
        payload={"msgtype": "text"},
        message_id="msg-late-callback-001",
    )
    callback = SimpleNamespace(data=late_message.data)
    handler = dingtalk_module._IncomingHandler(adapter, asyncio.get_running_loop())

    monkeypatch.setattr(
        dingtalk_module,
        "AckMessage",
        SimpleNamespace(STATUS_OK="ACK_OK", STATUS_SYSTEM_EXCEPTION="ACK_ERROR"),
    )
    monkeypatch.setattr(
        dingtalk_module,
        "ChatbotMessage",
        SimpleNamespace(from_dict=lambda _payload: late_message),
        raising=False,
    )
    disconnect_task = asyncio.create_task(adapter.disconnect())
    await asyncio.sleep(0)

    ack = await handler.process(callback)

    await disconnect_task
    await asyncio.sleep(0)

    assert ack == ("ACK_OK", "OK")
    websocket_close.assert_awaited_once()
    assert inflight_cancelled.is_set() is True
    assert relay_calls == ["msg-inflight-001"]
    adapter.handle_message.assert_not_called()
    assert len(adapter._xintai_relay_tasks) == 0


@pytest.mark.asyncio
async def test_direct_late_on_message_is_dropped_once_disconnect_begins() -> None:
    adapter = _make_connected_adapter()
    adapter.handle_message = AsyncMock()
    adapter._http_client = AsyncMock()
    adapter._stream_client = SimpleNamespace(stop=AsyncMock())
    adapter._stream_task = asyncio.create_task(asyncio.sleep(60))
    adapter._xintai_evidence_relay = SimpleNamespace(
        relay=AsyncMock(return_value=RelayResult(True, "msg-direct-late-001", "accepted")),
        bind_http_client=lambda _client: None,
    )

    disconnect_task = asyncio.create_task(adapter.disconnect())
    await asyncio.sleep(0)

    late_message = _make_dingtalk_message(
        text="关停后别再回我",
        payload={"msgtype": "text"},
        message_id="msg-direct-late-001",
    )
    await adapter._on_message(late_message)

    await disconnect_task
    await asyncio.sleep(0)

    adapter._xintai_evidence_relay.relay.assert_not_awaited()
    adapter.handle_message.assert_not_called()
    assert adapter._xintai_stream_health.last_event_at is None
    assert len(adapter._xintai_relay_tasks) == 0


@pytest.mark.asyncio
async def test_stream_process_records_callback_proof_before_background_dispatch(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_connected_adapter()
    adapter._on_message = AsyncMock()
    callback_message = _make_dingtalk_message(
        text="敏感原文不要进账本",
        payload={
            "msgtype": "file",
            "fileName": "日报.xlsx",
            "downloadCode": "download-secret-001",
            "sessionWebhook": "https://callback.example/webhook",
        },
        message_id="msg-sensitive-001",
        sender_id="sender-sensitive-001",
        conversation_id="conversation-sensitive-001",
        conversation_type="2",
    )
    create_task_seen: list[str] = []
    proof_written_before_first_create_task = False

    def _fake_create_task(coro):
        nonlocal proof_written_before_first_create_task
        create_task_seen.append("create_task")
        if len(create_task_seen) == 1:
            assert get_proof_ledger_path().exists() is True
            entries = _read_callback_proof_entries()
            assert len(entries) == 1
            assert entries[0]["message_type"] == "file"
            assert entries[0]["channel_type"] == "group"
            proof_written_before_first_create_task = True
        coro.close()
        task = asyncio.get_running_loop().create_future()
        task.set_result(None)
        return task

    monkeypatch.setattr(
        dingtalk_module,
        "AckMessage",
        SimpleNamespace(STATUS_OK="ACK_OK", STATUS_SYSTEM_EXCEPTION="ACK_ERROR"),
    )
    monkeypatch.setattr(
        dingtalk_module,
        "ChatbotMessage",
        SimpleNamespace(from_dict=lambda _payload: callback_message),
        raising=False,
    )
    monkeypatch.setattr(dingtalk_module.asyncio, "create_task", _fake_create_task)

    result = await dingtalk_module._IncomingHandler(
        adapter,
        asyncio.get_running_loop(),
    ).process(SimpleNamespace(data=callback_message.data))

    assert result == ("ACK_OK", "OK")
    assert create_task_seen
    assert proof_written_before_first_create_task is True
    serialized = get_proof_ledger_path().read_text(encoding="utf-8")
    assert "敏感原文不要进账本" not in serialized
    assert "日报.xlsx" not in serialized
    assert "sender-sensitive-001" not in serialized
    assert "conversation-sensitive-001" not in serialized
    assert "msg-sensitive-001" not in serialized
    assert "download-secret-001" not in serialized
    assert "callback.example/webhook" not in serialized


@pytest.mark.asyncio
async def test_stream_process_empty_trace_skips_callback_proof_ledger(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_connected_adapter()
    adapter._on_message = AsyncMock()
    adapter._resolve_message_id = MagicMock(return_value="  ")
    callback_message = _make_dingtalk_message(
        text="没有 trace 也要继续处理",
        payload={"msgtype": "text", "messageId": None},
        message_id=None,
    )
    recorded_calls = []

    def _fake_create_task(coro):
        coro.close()
        task = asyncio.get_running_loop().create_future()
        task.set_result(None)
        return task

    monkeypatch.setattr(
        dingtalk_module,
        "AckMessage",
        SimpleNamespace(STATUS_OK="ACK_OK", STATUS_SYSTEM_EXCEPTION="ACK_ERROR"),
    )
    monkeypatch.setattr(
        dingtalk_module,
        "ChatbotMessage",
        SimpleNamespace(from_dict=lambda _payload: callback_message),
        raising=False,
    )
    monkeypatch.setattr(
        dingtalk_module,
        "record_stream_callback_proof",
        lambda **kwargs: recorded_calls.append(kwargs),
    )
    monkeypatch.setattr(dingtalk_module.asyncio, "create_task", _fake_create_task)

    result = await dingtalk_module._IncomingHandler(
        adapter,
        asyncio.get_running_loop(),
    ).process(SimpleNamespace(data=callback_message.data))

    assert result == ("ACK_OK", "OK")
    assert recorded_calls == []
    assert get_proof_ledger_path().exists() is False


@pytest.mark.asyncio
async def test_stream_process_passes_injected_utc_callback_receive_time(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_connected_adapter()
    adapter._on_message = AsyncMock()
    callback_message = _make_dingtalk_message(
        text="固定回调时间",
        payload={"msgtype": "text"},
        message_id="msg-fixed-clock-001",
    )
    captured = {}

    def _fake_create_task(coro):
        coro.close()
        task = asyncio.get_running_loop().create_future()
        task.set_result(None)
        return task

    monkeypatch.setattr(
        dingtalk_module,
        "AckMessage",
        SimpleNamespace(STATUS_OK="ACK_OK", STATUS_SYSTEM_EXCEPTION="ACK_ERROR"),
    )
    monkeypatch.setattr(
        dingtalk_module,
        "ChatbotMessage",
        SimpleNamespace(from_dict=lambda _payload: callback_message),
        raising=False,
    )
    monkeypatch.setattr(
        dingtalk_module,
        "record_stream_callback_proof",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(dingtalk_module.asyncio, "create_task", _fake_create_task)

    fixed_clock = lambda: datetime(2026, 7, 16, 8, 9, 10, tzinfo=timezone.utc)
    result = await dingtalk_module._IncomingHandler(
        adapter,
        asyncio.get_running_loop(),
        clock=fixed_clock,
    ).process(SimpleNamespace(data=callback_message.data))

    assert result == ("ACK_OK", "OK")
    assert captured["callback_receive_time"] == "2026-07-16T08:09:10+00:00"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("accepting_events", "shutting_down"),
    [
        (False, False),
        (True, True),
    ],
    ids=["inactive", "shutdown"],
)
async def test_stream_process_inactive_callback_still_records_ledger_without_dispatch(
    monkeypatch,
    tmp_path,
    accepting_events,
    shutting_down,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_connected_adapter()
    adapter._accepting_events = accepting_events
    adapter._shutting_down = shutting_down
    adapter._on_message = AsyncMock()
    adapter._spawn_bg = MagicMock()
    callback_message = _make_dingtalk_message(
        text="停机窗口也要留证明",
        payload={"msgtype": "text"},
        message_id="msg-inactive-proof-001",
        conversation_type="2",
    )
    create_task_seen: list[str] = []

    def _fake_create_task(coro):
        create_task_seen.append("create_task")
        coro.close()
        task = asyncio.get_running_loop().create_future()
        task.set_result(None)
        return task

    monkeypatch.setattr(
        dingtalk_module,
        "AckMessage",
        SimpleNamespace(STATUS_OK="ACK_OK", STATUS_SYSTEM_EXCEPTION="ACK_ERROR"),
    )
    monkeypatch.setattr(
        dingtalk_module,
        "ChatbotMessage",
        SimpleNamespace(from_dict=lambda _payload: callback_message),
        raising=False,
    )
    monkeypatch.setattr(dingtalk_module.asyncio, "create_task", _fake_create_task)

    result = await dingtalk_module._IncomingHandler(
        adapter,
        asyncio.get_running_loop(),
    ).process(SimpleNamespace(data=callback_message.data))

    assert result == ("ACK_OK", "OK")
    assert create_task_seen == []
    adapter._spawn_bg.assert_not_called()
    adapter._on_message.assert_not_called()
    entries = _read_callback_proof_entries()
    assert len(entries) == 1
    assert entries[0]["message_type"] == "text"
    assert entries[0]["channel_type"] == "group"


@pytest.mark.asyncio
async def test_stream_process_duplicate_callback_keeps_single_proof_entry(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_connected_adapter()
    adapter._on_message = AsyncMock()
    callback_message = _make_dingtalk_message(
        text="重复消息",
        payload={"msgtype": "text"},
        message_id="msg-duplicate-proof-001",
        conversation_type="1",
    )

    def _fake_create_task(coro):
        coro.close()
        task = asyncio.get_running_loop().create_future()
        task.set_result(None)
        return task

    monkeypatch.setattr(
        dingtalk_module,
        "AckMessage",
        SimpleNamespace(STATUS_OK="ACK_OK", STATUS_SYSTEM_EXCEPTION="ACK_ERROR"),
    )
    monkeypatch.setattr(
        dingtalk_module,
        "ChatbotMessage",
        SimpleNamespace(from_dict=lambda _payload: callback_message),
        raising=False,
    )
    monkeypatch.setattr(dingtalk_module.asyncio, "create_task", _fake_create_task)
    handler = dingtalk_module._IncomingHandler(adapter, asyncio.get_running_loop())
    callback = SimpleNamespace(data=callback_message.data)

    first = await handler.process(callback)
    second = await handler.process(callback)

    assert first == ("ACK_OK", "OK")
    assert second == ("ACK_OK", "OK")
    entries = _read_callback_proof_entries()
    assert len(entries) == 1
    assert entries[0]["message_type"] == "text"
    assert entries[0]["channel_type"] == "private"


@pytest.mark.asyncio
async def test_stream_process_ledger_failure_still_acks_and_schedules_background_work(
    monkeypatch,
    caplog,
) -> None:
    adapter = _make_connected_adapter()
    adapter._on_message = AsyncMock()
    callback_message = _make_dingtalk_message(
        text="账本失败也别挡住处理",
        payload={"msgtype": "text"},
        message_id="msg-ledger-error-001",
    )
    created_tasks = []

    def _boom(**_kwargs):
        raise RuntimeError("ledger exploded")

    monkeypatch.setattr(
        dingtalk_module,
        "AckMessage",
        SimpleNamespace(STATUS_OK="ACK_OK", STATUS_SYSTEM_EXCEPTION="ACK_ERROR"),
    )
    monkeypatch.setattr(
        dingtalk_module,
        "ChatbotMessage",
        SimpleNamespace(from_dict=lambda _payload: callback_message),
        raising=False,
    )
    monkeypatch.setattr(dingtalk_module, "record_stream_callback_proof", _boom)
    original_create_task = asyncio.create_task

    def _tracking_create_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(dingtalk_module.asyncio, "create_task", _tracking_create_task)
    handler = dingtalk_module._IncomingHandler(adapter, asyncio.get_running_loop())

    with caplog.at_level("ERROR"):
        ack = await handler.process(SimpleNamespace(data=callback_message.data))
        await asyncio.sleep(0)
        if created_tasks:
            await asyncio.gather(*created_tasks)

    assert ack == ("ACK_OK", "OK")
    adapter._on_message.assert_awaited_once()
    assert "proof ledger" in caplog.text.lower()
    assert "ledger exploded" in caplog.text


@pytest.mark.asyncio
async def test_direct_on_message_does_not_write_callback_proof_ledger(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_connected_adapter()
    adapter._xintai_evidence_relay = SimpleNamespace(
        relay=AsyncMock(return_value=RelayResult(True, "msg-direct-only-001", "accepted"))
    )
    adapter.handle_message = AsyncMock()
    direct_message = _make_dingtalk_message(
        text="直接调用 _on_message 不记账",
        payload={"msgtype": "text"},
        message_id="msg-direct-only-001",
    )

    await adapter._on_message(direct_message)
    tasks = list(adapter._xintai_relay_tasks)
    if tasks:
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)

    assert get_proof_ledger_path().exists() is False


def test_build_xintai_payload_uses_event_timestamp_metadata() -> None:
    adapter = DingTalkAdapter(PlatformConfig(enabled=True))
    message = _make_dingtalk_message(
        text="",
        payload={"msgtype": "attachment", "fileName": "日报.xlsx", "fileId": "file-001"},
        message_id="msg-time-meta-001",
        create_at="1720688400000",
    )

    payload = adapter._build_xintai_payload(
        message,
        text="",
        message_id="msg-time-meta-001",
        timestamp=datetime.fromtimestamp(1720688400, tz=timezone.utc),
    )

    assert payload["receivedAt"] == "2024-07-11T09:00:00+00:00"
    assert payload["received_at"] == "2024-07-11T09:00:00+00:00"
    assert payload["messageTime"] == "1720688400000"
    assert payload["msgCreateTime"] == "1720688400000"
    assert payload["createTime"] == "1720688400000"
    assert payload["eventTime"] == "1720688400000"


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DINGTALK: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.DINGTALK: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._update_prompt_pending = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()

    source = SessionSource(
        platform=Platform.DINGTALK,
        user_id="dt-user-001",
        user_name="测试员",
        chat_id="cid-dingtalk-001",
        chat_type="group",
    )
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-dingtalk-001",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DINGTALK,
        chat_type="group",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    return runner, source


@pytest.mark.asyncio
async def test_keyword_bridge_no_longer_intercepts_dingtalk_messages() -> None:
    runner, source = _make_runner()
    runner._handle_xt_data_center_dingtalk_command = AsyncMock(return_value="old-bridge")
    runner._handle_message_with_agent = AsyncMock(return_value="agent reply")

    result = await runner._handle_message(
        MessageEvent(text="产量怎么样", source=source, message_id="msg-keyword-001")
    )

    assert result == "agent reply"
    runner._handle_xt_data_center_dingtalk_command.assert_not_called()
    runner._handle_message_with_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_help_and_status_use_xintai_identity(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    monkeypatch.setattr(
        "gateway.slash_commands.t",
        lambda key, **kwargs: {
            "gateway.help.header": "📖 **鑫泰铝业智能大脑 可用指令**",
            "gateway.status.header": "📊 **鑫泰铝业智能大脑 运行状态**",
        }.get(key, key),
    )
    runner, source = _make_runner()

    help_result = await runner._handle_message(
        MessageEvent(text="/help", source=source, message_id="msg-help-001")
    )
    status_result = await runner._handle_message(
        MessageEvent(text="/status", source=source, message_id="msg-status-001")
    )

    assert help_result.startswith("📖 **鑫泰铝业智能大脑 可用指令**")
    assert "Hermes Commands" not in help_result
    assert "设置当前会话标题" in help_result
    assert "更新当前智能大脑程序" in help_result
    assert "Hermes Agent" not in help_result
    assert "Update Hermes Agent to the latest version" not in help_result
    assert "Update" not in help_result
    assert status_result.startswith("📊 **鑫泰铝业智能大脑 运行状态**")
    assert "Hermes Gateway Status" not in status_result


@pytest.mark.asyncio
async def test_status_exposes_dingtalk_stream_health_snapshot(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    monkeypatch.setattr(
        "gateway.slash_commands.t",
        lambda key, **kwargs: {
            "gateway.status.header": "📊 **鑫泰铝业智能大脑 运行状态**",
        }.get(key, key),
    )
    runner, source = _make_runner()
    runner.adapters[Platform.DINGTALK].get_xintai_stream_health.return_value = {
        "connected": True,
        "stream_running": True,
        "last_event_at": None,
        "relay_success_count": 3,
        "relay_failure_count": 1,
        "last_error": "relay_http_500",
    }

    result = await runner._handle_message(
        MessageEvent(text="/status", source=source, message_id="msg-status-health-001")
    )

    assert result.startswith("📊 **鑫泰铝业智能大脑 运行状态**")
    assert "**钉钉 Stream：** 运行中" in result
    assert "**最近收到事件：** 暂无" in result
    assert "**转发成功次数：** 3" in result
    assert "**转发失败次数：** 1" in result
    assert "**最近错误：** relay_http_500" in result


@pytest.mark.asyncio
async def test_missing_message_id_text_events_use_stable_idempotent_identity() -> None:
    adapter = _make_connected_adapter()
    adapter._xintai_evidence_relay = SimpleNamespace(
        relay=AsyncMock(return_value=RelayResult(True, "trace-text-001", "accepted"))
    )
    adapter.handle_message = AsyncMock()

    repeated_a = _make_dingtalk_message(
        text="昨天那个先继续跟一下",
        payload={"msgtype": "text", "messageId": None},
        message_id=None,
        create_at="1720688400000",
    )
    repeated_b = _make_dingtalk_message(
        text="昨天那个先继续跟一下",
        payload={"msgtype": "text", "messageId": None},
        message_id=None,
        create_at="1720688400000",
    )
    different = _make_dingtalk_message(
        text="今天这个改成新的",
        payload={"msgtype": "text", "messageId": None},
        message_id=None,
        create_at="1720688400000",
    )

    await adapter._on_message(repeated_a)
    await adapter._on_message(repeated_b)
    await adapter._on_message(different)
    tasks = list(adapter._xintai_relay_tasks)
    if tasks:
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)

    assert adapter._xintai_evidence_relay.relay.await_count == 2
    assert adapter.handle_message.await_count == 2

    first_payload = adapter._xintai_evidence_relay.relay.await_args_list[0].args[0]
    second_payload = adapter._xintai_evidence_relay.relay.await_args_list[1].args[0]
    first_event = adapter.handle_message.await_args_list[0].args[0]
    second_event = adapter.handle_message.await_args_list[1].args[0]

    assert first_payload["messageId"] == first_payload["traceId"] == first_event.message_id
    assert second_payload["messageId"] == second_payload["traceId"] == second_event.message_id
    assert first_event.message_id.startswith("dingtalk-stream-sha256:")
    assert first_event.message_id != second_event.message_id


@pytest.mark.asyncio
async def test_missing_message_id_file_events_use_stable_idempotent_identity() -> None:
    adapter = _make_connected_adapter()
    adapter._xintai_evidence_relay = SimpleNamespace(
        relay=AsyncMock(return_value=RelayResult(True, "trace-file-001", "accepted"))
    )
    adapter.handle_message = AsyncMock()

    repeated_a = _make_dingtalk_message(
        text="",
        payload={
            "msgtype": "file",
            "messageId": None,
            "fileName": "日报.xlsx",
            "fileId": "file-001",
            "downloadCode": "download-secret-001",
        },
        message_id=None,
        create_at="1720688400000",
    )
    repeated_b = _make_dingtalk_message(
        text="",
        payload={
            "msgtype": "file",
            "messageId": None,
            "fileName": "日报.xlsx",
            "fileId": "file-001",
            "downloadCode": "download-secret-002",
        },
        message_id=None,
        create_at="1720688400000",
    )
    different = _make_dingtalk_message(
        text="",
        payload={
            "msgtype": "file",
            "messageId": None,
            "fileName": "日报-新.xlsx",
            "fileId": "file-002",
            "downloadCode": "download-secret-003",
        },
        message_id=None,
        create_at="1720688400000",
    )

    await adapter._on_message(repeated_a)
    await adapter._on_message(repeated_b)
    await adapter._on_message(different)
    tasks = list(adapter._xintai_relay_tasks)
    if tasks:
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)

    assert adapter._xintai_evidence_relay.relay.await_count == 2
    adapter.handle_message.assert_not_called()

    first_payload = adapter._xintai_evidence_relay.relay.await_args_list[0].args[0]
    second_payload = adapter._xintai_evidence_relay.relay.await_args_list[1].args[0]

    assert first_payload["messageId"] == first_payload["traceId"]
    assert second_payload["messageId"] == second_payload["traceId"]
    assert first_payload["messageId"].startswith("dingtalk-stream-sha256:")
    assert first_payload["messageId"] != second_payload["messageId"]
    assert "download-secret-001" not in first_payload["messageId"]
    assert "download-secret-002" not in first_payload["messageId"]


def test_runtime_soul_sync_keeps_xintai_identity_in_chinese(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = sync_xintai_runtime_soul()
    soul = (tmp_path / "SOUL.md").read_text(encoding="utf-8")

    assert result.installed is True
    assert "鑫泰铝业智能大脑" in soul
    assert "只用中文" in soul
    assert "钉钉证据" in soul
    assert "MES/WMS" in soul
    assert "只读数据" in soul
    assert "数据中枢已确认事实" in soul
    assert "不要把自己说成开发助手" in soul
