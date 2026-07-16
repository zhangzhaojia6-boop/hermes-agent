from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json

from gateway import xintai_callback_proof_ledger as ledger
import pytest


def _ledger_file(tmp_path):
    return tmp_path / "gateway" / "xintai_callback_proof_ledger.json"


def test_get_proof_ledger_path_tracks_active_hermes_home(monkeypatch, tmp_path) -> None:
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    path_a = ledger.get_proof_ledger_path()

    monkeypatch.setenv("HERMES_HOME", str(home_b))
    path_b = ledger.get_proof_ledger_path()

    assert path_a == home_a / "gateway" / "xintai_callback_proof_ledger.json"
    assert path_b == home_b / "gateway" / "xintai_callback_proof_ledger.json"
    assert path_a != path_b


def test_load_ledger_entries_rejects_bare_list_and_accepts_entries_wrapper(
    tmp_path,
    caplog,
) -> None:
    ledger_path = _ledger_file(tmp_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    bare_entry = {
        "trace_hash": "a" * 64,
        "message_type": "text",
        "channel_type": "group",
        "callback_receive_time": "2026-07-16T01:02:03+00:00",
        "source": "stream_callback",
    }
    wrapped_entry = {
        "trace_hash": "b" * 64,
        "message_type": "file",
        "channel_type": "private",
        "callback_receive_time": "2026-07-16T02:03:04+00:00",
        "source": "stream_callback",
    }

    ledger_path.write_text(json.dumps([bare_entry]), encoding="utf-8")
    assert ledger._load_ledger_entries(ledger_path) == []
    assert "invalid top-level shape" in caplog.text.lower()

    ledger_path.write_text(json.dumps({"entries": [wrapped_entry]}), encoding="utf-8")
    assert ledger._load_ledger_entries(ledger_path) == [wrapped_entry]


@pytest.mark.parametrize(
    ("bad_entry", "warning_fragment"),
    [
        (
            {
                "trace_hash": "not-a-sha",
                "message_type": "text",
                "channel_type": "group",
                "callback_receive_time": "2026-07-16T01:02:03+00:00",
                "source": "stream_callback",
            },
            "dropped malformed entries",
        ),
        (
            {
                "trace_hash": "a" * 64,
                "message_type": "text",
                "channel_type": "room",
                "callback_receive_time": "2026-07-16T01:02:03+00:00",
                "source": "stream_callback",
            },
            "dropped malformed entries",
        ),
        (
            {
                "trace_hash": "a" * 64,
                "message_type": "x" * 129,
                "channel_type": "group",
                "callback_receive_time": "2026-07-16T01:02:03+00:00",
                "source": "stream_callback",
            },
            "dropped malformed entries",
        ),
        (
            {
                "trace_hash": "a" * 64,
                "message_type": "text",
                "channel_type": "group",
                "callback_receive_time": "x" * 65,
                "source": "stream_callback",
            },
            "dropped malformed entries",
        ),
        (
            {
                "trace_hash": "a" * 64,
                "message_type": "text",
                "channel_type": "group",
                "callback_receive_time": "2026-07-16T01:02:03+00:00",
                "source": "other",
            },
            "dropped malformed entries",
        ),
    ],
    ids=[
        "invalid-hash",
        "invalid-channel",
        "too-long-message-type",
        "too-long-callback-time",
        "invalid-source",
    ],
)
def test_load_ledger_entries_drops_invalid_entry_variants(tmp_path, caplog, bad_entry, warning_fragment) -> None:
    ledger_path = _ledger_file(tmp_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    valid_entry = {
        "trace_hash": "b" * 64,
        "message_type": "text",
        "channel_type": "private",
        "callback_receive_time": "2026-07-16T02:03:04+00:00",
        "source": "stream_callback",
    }

    ledger_path.write_text(json.dumps({"entries": [bad_entry, valid_entry]}), encoding="utf-8")

    assert ledger._load_ledger_entries(ledger_path) == [valid_entry]
    assert warning_fragment in caplog.text.lower()


def test_record_stream_callback_proof_writes_strict_fields_and_hash(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    ledger.record_stream_callback_proof(
        trace_id="trace-001",
        message_type="picture" * 30,
        channel_type="group",
        callback_receive_time="2026-07-16T03:04:05+00:00",
    )

    payload = json.loads(_ledger_file(tmp_path).read_text(encoding="utf-8"))
    assert list(payload) == ["entries"]
    assert len(payload["entries"]) == 1
    entry = payload["entries"][0]
    assert set(entry) == {
        "trace_hash",
        "message_type",
        "channel_type",
        "callback_receive_time",
        "source",
    }
    assert entry == {
        "trace_hash": sha256(b"trace-001").hexdigest(),
        "message_type": ("picture" * 30)[:128],
        "channel_type": "group",
        "callback_receive_time": "2026-07-16T03:04:05+00:00",
        "source": "stream_callback",
    }
    serialized = _ledger_file(tmp_path).read_text(encoding="utf-8")
    assert "trace-001" not in serialized
    assert "conversation" not in serialized
    assert "sender" not in serialized
    assert "downloadCode" not in serialized


def test_record_stream_callback_proof_deduplicates_by_trace_hash(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    ledger.record_stream_callback_proof(
        trace_id="trace-duplicate",
        message_type="text",
        channel_type="group",
        callback_receive_time="2026-07-16T04:05:06+00:00",
    )
    ledger.record_stream_callback_proof(
        trace_id="trace-duplicate",
        message_type="file",
        channel_type="private",
        callback_receive_time="2026-07-16T07:08:09+00:00",
    )

    payload = json.loads(_ledger_file(tmp_path).read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["message_type"] == "text"
    assert payload["entries"][0]["channel_type"] == "group"


def test_record_stream_callback_proof_skips_blank_trace_id(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    ledger.record_stream_callback_proof(
        trace_id="   ",
        message_type="text",
        channel_type="group",
        callback_receive_time="2026-07-16T04:05:06+00:00",
    )

    assert _ledger_file(tmp_path).exists() is False


def test_record_stream_callback_proof_keeps_latest_2000_entries(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    existing_entries = [
        {
            "trace_hash": f"{idx:064x}",
            "message_type": "text",
            "channel_type": "group",
            "callback_receive_time": f"2026-07-16T00:00:{idx % 60:02d}+00:00",
            "source": "stream_callback",
        }
        for idx in range(2000)
    ]
    _ledger_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _ledger_file(tmp_path).write_text(
        json.dumps({"entries": existing_entries}),
        encoding="utf-8",
    )

    ledger.record_stream_callback_proof(
        trace_id="trace-newest",
        message_type="image",
        channel_type="private",
        callback_receive_time="2026-07-16T09:10:11+00:00",
    )

    payload = json.loads(_ledger_file(tmp_path).read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 2000
    assert payload["entries"][0]["trace_hash"] == f"{1:064x}"
    assert payload["entries"][-1]["trace_hash"] == sha256(b"trace-newest").hexdigest()


def test_record_stream_callback_proof_duplicate_rewrites_trimmed_ledger(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    duplicate_hash = sha256(b"trace-duplicate-existing").hexdigest()
    existing_entries = [
        {
            "trace_hash": f"{idx:064x}",
            "message_type": "text",
            "channel_type": "group",
            "callback_receive_time": f"2026-07-16T00:00:{idx % 60:02d}+00:00",
            "source": "stream_callback",
        }
        for idx in range(2000)
    ]
    existing_entries.append(
        {
            "trace_hash": duplicate_hash,
            "message_type": "text",
            "channel_type": "private",
            "callback_receive_time": "2026-07-16T09:10:11+00:00",
            "source": "stream_callback",
        }
    )
    _ledger_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _ledger_file(tmp_path).write_text(json.dumps({"entries": existing_entries}), encoding="utf-8")

    ledger.record_stream_callback_proof(
        trace_id="trace-duplicate-existing",
        message_type="ignored",
        channel_type="group",
        callback_receive_time="2026-07-16T10:11:12+00:00",
    )

    payload = json.loads(_ledger_file(tmp_path).read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 2000
    assert payload["entries"][0]["trace_hash"] == f"{1:064x}"
    assert payload["entries"][-1]["trace_hash"] == duplicate_hash


def test_record_stream_callback_proof_recovers_from_corrupted_file(monkeypatch, tmp_path, caplog) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _ledger_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _ledger_file(tmp_path).write_text("{not-json", encoding="utf-8")

    ledger.record_stream_callback_proof(
        trace_id="trace-recover",
        message_type="file",
        channel_type="group",
        callback_receive_time="2026-07-16T12:13:14+00:00",
    )

    payload = json.loads(_ledger_file(tmp_path).read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["trace_hash"] == sha256(b"trace-recover").hexdigest()
    assert "proof ledger" in caplog.text.lower()


def test_record_stream_callback_proof_recovers_from_oversized_file(monkeypatch, tmp_path, caplog) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _ledger_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _ledger_file(tmp_path).write_text("x" * ((1024 * 1024) + 1), encoding="utf-8")

    ledger.record_stream_callback_proof(
        trace_id="trace-recover-oversized",
        message_type="text",
        channel_type="group",
        callback_receive_time="2026-07-16T12:13:14+00:00",
    )

    payload = json.loads(_ledger_file(tmp_path).read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["trace_hash"] == sha256(b"trace-recover-oversized").hexdigest()
    assert "oversized" in caplog.text.lower()


def test_record_stream_callback_proof_recovers_from_non_list_entries(monkeypatch, tmp_path, caplog) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _ledger_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _ledger_file(tmp_path).write_text(json.dumps({"entries": {"bad": "shape"}}), encoding="utf-8")

    ledger.record_stream_callback_proof(
        trace_id="trace-recover-from-dict",
        message_type="text",
        channel_type="private",
        callback_receive_time="2026-07-16T12:13:14+00:00",
    )

    payload = json.loads(_ledger_file(tmp_path).read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["trace_hash"] == sha256(b"trace-recover-from-dict").hexdigest()
    assert "invalid top-level shape" in caplog.text.lower()


def test_record_stream_callback_proof_uses_atomic_json_write(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    captured = {}

    def _fake_atomic_json_write(path, payload, **kwargs):
        captured["path"] = path
        captured["payload"] = payload
        captured["kwargs"] = kwargs

    monkeypatch.setattr(ledger, "atomic_json_write", _fake_atomic_json_write)

    ledger.record_stream_callback_proof(
        trace_id="trace-atomic",
        message_type="text",
        channel_type="private",
        callback_receive_time="2026-07-16T15:16:17+00:00",
    )

    assert captured["path"] == _ledger_file(tmp_path)
    assert list(captured["payload"]) == ["entries"]
    assert captured["payload"]["entries"][0]["trace_hash"] == sha256(b"trace-atomic").hexdigest()
    assert captured["kwargs"]["mode"] == 0o600


def test_record_stream_callback_proof_is_thread_safe(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _record(idx: int) -> None:
        ledger.record_stream_callback_proof(
            trace_id=f"trace-thread-{idx:02d}",
            message_type=f"text-{idx}",
            channel_type="group" if idx % 2 == 0 else "private",
            callback_receive_time=f"2026-07-16T15:16:{idx:02d}+00:00",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_record, range(20)))

    payload = json.loads(_ledger_file(tmp_path).read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 20
    assert {entry["trace_hash"] for entry in payload["entries"]} == {
        sha256(f"trace-thread-{idx:02d}".encode("utf-8")).hexdigest()
        for idx in range(20)
    }
