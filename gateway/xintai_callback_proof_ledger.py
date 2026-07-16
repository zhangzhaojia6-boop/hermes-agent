"""Privacy-safe proof ledger for DingTalk stream callbacks.

This ledger only proves that the SDK callback boundary received an event.
It is not replay, delivery, or business-processing authority.
"""

from __future__ import annotations

from hashlib import sha256
import json
import logging
from pathlib import Path
import threading
from typing import cast

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_LEDGER_LOCK = threading.Lock()
_MAX_LEDGER_BYTES = 1024 * 1024
_MAX_ENTRIES = 2000
_TRACE_HASH_LENGTH = 64
_MESSAGE_TYPE_MAX = 128
_CALLBACK_RECEIVE_TIME_MAX = 64
_ENTRY_KEYS = (
    "trace_hash",
    "message_type",
    "channel_type",
    "callback_receive_time",
    "source",
)
_SOURCE = "stream_callback"


def get_proof_ledger_path() -> Path:
    return get_hermes_home() / "gateway" / "xintai_callback_proof_ledger.json"


def _is_lower_hex_digest(value: str) -> bool:
    return len(value) == _TRACE_HASH_LENGTH and all(ch in "0123456789abcdef" for ch in value)


def _normalize_message_type(value: object) -> str:
    normalized = str(value or "").strip() or "unknown"
    return normalized[:_MESSAGE_TYPE_MAX]


def _normalize_channel_type(value: object) -> str:
    return "group" if str(value or "").strip().lower() == "group" else "private"


def _normalize_callback_receive_time(value: object) -> str:
    return str(value or "").strip()[:_CALLBACK_RECEIVE_TIME_MAX]


def _is_valid_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    if tuple(entry.keys()) != _ENTRY_KEYS and set(entry.keys()) != set(_ENTRY_KEYS):
        return False
    typed_entry = cast(dict[str, object], entry)
    trace_hash = typed_entry.get("trace_hash")
    message_type = typed_entry.get("message_type")
    channel_type = typed_entry.get("channel_type")
    callback_receive_time = typed_entry.get("callback_receive_time")
    source = typed_entry.get("source")
    return (
        isinstance(trace_hash, str)
        and _is_lower_hex_digest(trace_hash)
        and isinstance(message_type, str)
        and 0 < len(message_type) <= _MESSAGE_TYPE_MAX
        and isinstance(channel_type, str)
        and channel_type in {"group", "private"}
        and isinstance(callback_receive_time, str)
        and 0 < len(callback_receive_time) <= _CALLBACK_RECEIVE_TIME_MAX
        and isinstance(source, str)
        and source == _SOURCE
    )


def _load_ledger_state(path: Path | None = None) -> tuple[list[dict[str, str]], bool]:
    ledger_path = path or get_proof_ledger_path()
    if not ledger_path.exists():
        return [], False
    try:
        if ledger_path.stat().st_size > _MAX_LEDGER_BYTES:
            logger.warning(
                "Xintai callback proof ledger oversized at %s; resetting empty",
                ledger_path,
            )
            return [], False
    except OSError as exc:
        logger.warning(
            "Xintai callback proof ledger unreadable at %s; resetting empty (%s)",
            ledger_path,
            type(exc).__name__,
        )
        return [], False
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Xintai callback proof ledger unreadable at %s; resetting empty (%s)",
            ledger_path,
            type(exc).__name__,
        )
        return [], False

    if not isinstance(payload, dict):
        logger.warning(
            "Xintai callback proof ledger has invalid top-level shape at %s; resetting empty",
            ledger_path,
        )
        return [], False

    payload = payload.get("entries")
    if not isinstance(payload, list):
        logger.warning(
            "Xintai callback proof ledger has invalid top-level shape at %s; resetting empty",
            ledger_path,
        )
        return [], False

    entries = [entry for entry in payload if _is_valid_entry(entry)]
    rewrite_needed = len(entries) != len(payload)
    if len(entries) != len(payload):
        logger.warning(
            "Xintai callback proof ledger dropped malformed entries at %s",
            ledger_path,
        )
    deduped_entries: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    for entry in reversed(entries):
        trace_hash = entry["trace_hash"]
        if trace_hash in seen_hashes:
            rewrite_needed = True
            continue
        seen_hashes.add(trace_hash)
        deduped_entries.append(entry)
    entries = list(reversed(deduped_entries))
    if len(entries) > _MAX_ENTRIES:
        entries = entries[-_MAX_ENTRIES:]
        rewrite_needed = True
    return entries, rewrite_needed


def _load_ledger_entries(path: Path | None = None) -> list[dict[str, str]]:
    entries, _ = _load_ledger_state(path)
    return entries


def _write_ledger_entries(entries: list[dict[str, str]], path: Path | None = None) -> None:
    atomic_json_write(path or get_proof_ledger_path(), {"entries": entries}, mode=0o600)


def record_stream_callback_proof(
    *,
    trace_id: str,
    message_type: str,
    channel_type: str,
    callback_receive_time: str,
) -> None:
    normalized_trace = trace_id.strip()
    if not normalized_trace:
        return

    ledger_path = get_proof_ledger_path()
    trace_hash = sha256(normalized_trace.encode("utf-8")).hexdigest()
    with _LEDGER_LOCK:
        entries, rewrite_needed = _load_ledger_state(ledger_path)
        if any(entry["trace_hash"] == trace_hash for entry in entries):
            if rewrite_needed:
                _write_ledger_entries(entries, ledger_path)
            return

        entries.append(
            {
                "trace_hash": trace_hash,
                "message_type": _normalize_message_type(message_type),
                "channel_type": _normalize_channel_type(channel_type),
                "callback_receive_time": _normalize_callback_receive_time(callback_receive_time),
                "source": _SOURCE,
            }
        )
        if len(entries) > _MAX_ENTRIES:
            entries = entries[-_MAX_ENTRIES:]

        _write_ledger_entries(entries, ledger_path)
