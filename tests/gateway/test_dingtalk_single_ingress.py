from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.platforms.dingtalk import DingTalkAdapter
from gateway.platforms import dingtalk as dingtalk_module
from gateway.session import SessionEntry, SessionSource, build_session_key
from gateway.xintai_evidence_relay import RelayResult


def _make_dingtalk_message(*, text: str = "", payload: dict | None = None, **overrides):
    payload = dict(payload or {})
    msg = MagicMock()
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
    assert relay_payload["rawEvent"]["eventType"] == "robot_notice"
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
    stop_called = asyncio.Event()
    relay_calls: list[str] = []
    submitted: list[str] = []

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

    async def _stop_stream():
        stop_called.set()

    def _submit(coro, loop):
        task = loop.create_task(coro)
        submitted.append(task.get_coro().cr_code.co_name)
        future = concurrent.futures.Future()

        def _copy_result(done: asyncio.Task) -> None:
            try:
                future.set_result(done.result())
            except Exception as exc:  # noqa: BLE001
                future.set_exception(exc)

        task.add_done_callback(_copy_result)
        return future

    adapter._xintai_evidence_relay = SimpleNamespace(relay=_relay, bind_http_client=lambda _client: None)
    adapter._stream_client = SimpleNamespace(stop=_stop_stream)
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
        "dingtalk_stream",
        SimpleNamespace(AckMessage=SimpleNamespace(STATUS_OK="ACK_OK")),
    )
    monkeypatch.setattr(
        dingtalk_module,
        "ChatbotMessage",
        SimpleNamespace(from_dict=lambda _payload: late_message),
        raising=False,
    )
    monkeypatch.setattr(dingtalk_module.asyncio, "run_coroutine_threadsafe", _submit)

    disconnect_task = asyncio.create_task(adapter.disconnect())
    await asyncio.sleep(0)

    ack = handler.process(callback)

    await disconnect_task
    await asyncio.sleep(0)

    assert ack == ("ACK_OK", "OK")
    assert stop_called.is_set() is True
    assert inflight_cancelled.is_set() is True
    assert relay_calls == ["msg-inflight-001"]
    assert submitted == []
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
async def test_runtime_ping_help_and_status_use_xintai_identity() -> None:
    runner, source = _make_runner()

    ping_result = await runner._handle_message(
        MessageEvent(text="/ping", source=source, message_id="msg-ping-001")
    )
    help_result = await runner._handle_message(
        MessageEvent(text="/help", source=source, message_id="msg-help-001")
    )
    status_result = await runner._handle_message(
        MessageEvent(text="/status", source=source, message_id="msg-status-001")
    )

    assert ping_result == "已收到。\n鑫泰铝业智能大脑网关已连接。"
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
async def test_status_exposes_dingtalk_stream_health_snapshot() -> None:
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

    assert "**钉钉 Stream Worker：** 运行中" in result
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


def test_docker_soul_declares_xintai_identity_in_chinese() -> None:
    soul = (Path(__file__).resolve().parents[2] / "docker" / "SOUL.md").read_text(encoding="utf-8")

    assert "鑫泰铝业智能大脑" in soul
    assert "只用中文" in soul
    assert "钉钉证据" in soul
    assert "MES/WMS只读数据" in soul
    assert "数据中枢已确认事实" in soul
    assert "不要把自己说成开发助手" in soul
