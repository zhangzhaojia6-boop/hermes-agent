from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
import logging
import os
import re
import time
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import httpx

from agent.redact import redact_sensitive_text


LOGGER = logging.getLogger(__name__)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(downloadcode|download_code|access_token|authorization|signature|sign|sig)\b\s*[:=]\s*[^&\s,;]+"
)
_SECRET_QUERY_KEYS = {"downloadcode", "download_code", "access_token", "authorization", "signature", "sign", "sig"}
_INLINE_FILE_BYTES_LIMIT = 10 * 1024 * 1024
_INLINE_FILE_RAW_KEYS = {
    "filecontentbase64",
    "file_content_base64",
    "filebytesbase64",
    "file_bytes_base64",
    "contentbase64",
    "filebytes",
    "file_bytes",
    "downloadedfilebytes",
}
_IDENTITY_EXCLUDED_KEYS = {
    "traceid",
    "trace_id",
    "receivedat",
    "received_at",
    "messageid",
    "message_id",
    "msgid",
    "msg_id",
}


@dataclass(frozen=True, slots=True)
class RelayResult:
    accepted: bool
    trace_id: str
    category: str
    error: str | None = None
    retry_after_seconds: float | None = None


class XintaiEvidenceRelay:
    def __init__(
        self,
        *,
        enabled: bool | None = None,
        base_url: str | None = None,
        inbound_token: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        request_timeout: float = 8.0,
        max_attempts: int = 7,
        retry_backoff_seconds: Sequence[float] = (1.0, 5.0, 15.0, 30.0, 60.0, 60.0),
    ) -> None:
        self.enabled = _env_bool("XINTAI_EVIDENCE_RELAY_ENABLED", False) if enabled is None else bool(enabled)
        self.base_url = str(base_url if base_url is not None else os.getenv("XINTAI_DATAHUB_BASE_URL", "")).strip()
        self.inbound_token = str(
            inbound_token if inbound_token is not None else os.getenv("XINTAI_DINGTALK_STREAM_RELAY_TOKEN", "")
        ).strip()
        self._http_client = http_client
        self.request_timeout = float(request_timeout)
        self.max_attempts = max(1, int(max_attempts))
        self.retry_backoff_seconds = tuple(float(item) for item in retry_backoff_seconds) or (0.25, 0.5, 1.0)

    def bind_http_client(self, client: httpx.AsyncClient | None) -> None:
        self._http_client = client

    async def relay(self, payload: Mapping[str, Any]) -> RelayResult:
        trace_id = _stable_trace_id(payload)
        normalized_payload = _normalize_payload(payload, trace_id=trace_id)
        if not self.enabled:
            return RelayResult(accepted=False, trace_id=trace_id, category="disabled")
        if not self.base_url or not self.inbound_token:
            return RelayResult(accepted=False, trace_id=trace_id, category="misconfigured")

        client = self._http_client or httpx.AsyncClient()
        created_client = self._http_client is None
        try:
            return await self._relay_with_client(client, normalized_payload, trace_id=trace_id)
        finally:
            if created_client:
                await client.aclose()

    async def _relay_with_client(
        self,
        client: httpx.AsyncClient,
        payload: Mapping[str, Any],
        *,
        trace_id: str,
    ) -> RelayResult:
        url = self.base_url.rstrip("/") + "/api/v1/dingtalk/agent-inbound"
        last_result = RelayResult(accepted=False, trace_id=trace_id, category="relay_not_attempted")

        for attempt in range(self.max_attempts):
            headers = _build_request_auth_headers(self.inbound_token, payload)
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=dict(payload),
                    timeout=self.request_timeout,
                )
            except httpx.TimeoutException as exc:
                last_result = RelayResult(
                    accepted=False,
                    trace_id=trace_id,
                    category="timeout",
                    error=_sanitize_error_text(str(exc), secrets=(self.inbound_token,)),
                )
            except httpx.HTTPError as exc:
                last_result = RelayResult(
                    accepted=False,
                    trace_id=trace_id,
                    category="network_error",
                    error=_sanitize_error_text(str(exc), secrets=(self.inbound_token,)),
                )
            except Exception as exc:  # noqa: BLE001
                last_result = RelayResult(
                    accepted=False,
                    trace_id=trace_id,
                    category="exception",
                    error=_sanitize_error_text(str(exc), secrets=(self.inbound_token,)),
                )
            else:
                last_result = self._response_result(response, trace_id=trace_id)
                if last_result.accepted:
                    return last_result

            if attempt >= self.max_attempts - 1 or not _is_retryable(last_result):
                break
            LOGGER.warning(
                "Xintai evidence relay retry trace_id=%s attempt=%d category=%s",
                trace_id,
                attempt + 1,
                last_result.category,
            )
            delay = last_result.retry_after_seconds
            if delay is None:
                delay = self.retry_backoff_seconds[min(attempt, len(self.retry_backoff_seconds) - 1)]
            await asyncio.sleep(max(0.0, min(float(delay), 120.0)))

        LOGGER.warning(
            "Xintai evidence relay failed trace_id=%s category=%s error=%s",
            trace_id,
            last_result.category,
            last_result.error or "",
        )
        return last_result

    def _response_result(self, response: httpx.Response | Any, *, trace_id: str) -> RelayResult:
        payload = _response_json(response)
        response_trace_id = _first_text(payload.get("trace_id"), payload.get("traceId")) or trace_id
        if 200 <= int(response.status_code) < 300:
            errcode = payload.get("errcode", 0)
            if errcode == 0:
                return RelayResult(accepted=True, trace_id=response_trace_id, category="accepted")
            error_text = _sanitize_error_text(
                _first_text(payload.get("errmsg"), payload.get("detail")) or getattr(response, "text", ""),
                secrets=(self.inbound_token,),
            )
            return RelayResult(
                accepted=False,
                trace_id=response_trace_id,
                category=f"errcode_{errcode}",
                error=error_text or None,
            )

        error_text = _sanitize_error_text(
            getattr(response, "text", "") or json.dumps(payload, ensure_ascii=False),
            secrets=(self.inbound_token,),
        )
        return RelayResult(
            accepted=False,
            trace_id=response_trace_id,
            category=f"http_{int(response.status_code)}",
            error=error_text or None,
            retry_after_seconds=_retry_after_seconds(response),
        )


def _build_request_auth_headers(secret: str, payload: Mapping[str, Any]) -> dict[str, str]:
    timestamp = str(int(time.time()))
    nonce = uuid4().hex
    kind = "dingtalk_stream"
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signed = (
        timestamp.encode("ascii")
        + b"."
        + nonce.encode("ascii")
        + b"."
        + kind.encode("ascii")
        + b"."
        + canonical
    )
    signature = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    return {
        "x-dingtalk-inbound-timestamp": timestamp,
        "x-dingtalk-inbound-nonce": nonce,
        "x-dingtalk-inbound-kind": kind,
        "x-dingtalk-inbound-signature": f"sha256={signature}",
    }


def _is_retryable(result: RelayResult) -> bool:
    if result.category in {"timeout", "network_error"}:
        return True
    if not result.category.startswith("http_"):
        return False
    try:
        status_code = int(result.category.removeprefix("http_"))
    except ValueError:
        return False
    return status_code in {408, 425, 429} or status_code >= 500


def _retry_after_seconds(response: Any) -> float | None:
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw_value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        value = float(str(raw_value).strip())
    except (TypeError, ValueError):
        return None
    return max(0.0, min(value, 120.0))


def _normalize_payload(payload: Mapping[str, Any], *, trace_id: str) -> dict[str, Any]:
    raw = dict(payload)
    message_id = _first_text(
        raw.get("messageId"),
        raw.get("msgId"),
        raw.get("message_id"),
        raw.get("messageid"),
        raw.get("fileId"),
        raw.get("mediaId"),
    ) or trace_id
    text_content = _extract_message_text(raw)
    message_type = _first_text(
        raw.get("msgtype"),
        raw.get("messageType"),
        raw.get("message_type"),
        raw.get("type"),
    ) or ("text" if text_content else "")
    normalized: dict[str, Any] = {
        "conversationId": _first_text(
            raw.get("conversationId"),
            raw.get("conversation_id"),
            raw.get("chatId"),
            raw.get("openConversationId"),
        ),
        "conversationType": _first_text(
            raw.get("conversationType"),
            raw.get("conversation_type"),
            raw.get("chatType"),
            raw.get("chat_type"),
        ),
        "senderStaffId": _first_text(
            raw.get("senderStaffId"),
            raw.get("sender_staff_id"),
            raw.get("senderId"),
            raw.get("senderUserId"),
            raw.get("userid"),
            raw.get("userId"),
        ),
        "senderId": _first_text(
            raw.get("senderId"),
            raw.get("sender_id"),
            raw.get("senderStaffId"),
            raw.get("senderUserId"),
        ),
        "senderUnionId": _first_text(raw.get("senderUnionId"), raw.get("sender_union_id"), raw.get("unionId")),
        "senderNick": _first_text(raw.get("senderNick"), raw.get("sender_nick")),
        "conversationTitle": _first_text(raw.get("conversationTitle"), raw.get("conversation_title")),
        "messageId": message_id,
        "msgId": message_id,
        "traceId": trace_id,
        "msgtype": message_type,
        "messageType": message_type,
        "createAt": _first_text(
            raw.get("createAt"),
            raw.get("create_at"),
            raw.get("messageTime"),
            raw.get("eventTime"),
            raw.get("event_time"),
            raw.get("timestamp"),
        ),
        "eventTime": _first_text(
            raw.get("eventTime"),
            raw.get("event_time"),
            raw.get("createAt"),
            raw.get("create_at"),
            raw.get("messageTime"),
            raw.get("timestamp"),
        ),
    }
    if text_content:
        normalized["text"] = {"content": text_content}
    file_name = _first_text(
        raw.get("fileName"),
        raw.get("file_name"),
        _path_value(raw, "content", "fileName"),
        _path_value(raw, "file", "fileName"),
    )
    if file_name:
        normalized["fileName"] = file_name
        normalized["file_name"] = file_name
    file_id = _first_text(
        raw.get("fileId"),
        raw.get("file_id"),
        raw.get("mediaId"),
        raw.get("media_id"),
        _path_value(raw, "content", "fileId"),
        _path_value(raw, "content", "mediaId"),
        _path_value(raw, "file", "fileId"),
        _path_value(raw, "file", "mediaId"),
    )
    if file_id:
        normalized["fileId"] = file_id
        normalized["file_id"] = file_id
        normalized["mediaId"] = file_id
    download_code = _first_text(
        raw.get("downloadCode"),
        raw.get("download_code"),
        _path_value(raw, "content", "downloadCode"),
        _path_value(raw, "file", "downloadCode"),
    )
    if download_code:
        normalized["downloadCode"] = download_code
    inline_content = _extract_inline_file_content(raw)
    if inline_content:
        normalized["fileContentBase64"] = inline_content["base64"]
        normalized["fileHash"] = inline_content["sha256"]
        normalized["file_hash"] = inline_content["sha256"]
    for date_key in ("business_date", "businessDate", "date", "reportDate"):
        if raw.get(date_key) not in (None, ""):
            normalized[date_key] = raw.get(date_key)
    safe_raw = _sanitize_raw_event(raw)
    normalized["rawEvent"] = safe_raw
    normalized["raw_event"] = safe_raw
    return {key: value for key, value in normalized.items() if value not in (None, "")}


def _extract_inline_file_content(payload: Mapping[str, Any]) -> dict[str, str] | None:
    base64_text = _first_text(
        payload.get("fileContentBase64"),
        payload.get("file_content_base64"),
        payload.get("fileBytesBase64"),
        payload.get("file_bytes_base64"),
        payload.get("contentBase64"),
    )
    if base64_text:
        if len(base64_text) > ((_INLINE_FILE_BYTES_LIMIT + 2) // 3) * 4 + 16:
            return None
        try:
            content = base64.b64decode(base64_text, validate=True)
        except (ValueError, TypeError):
            return None
        if len(content) > _INLINE_FILE_BYTES_LIMIT:
            return None
        return {"base64": base64_text, "sha256": hashlib.sha256(content).hexdigest()}

    raw_bytes = payload.get("fileBytes") or payload.get("file_bytes") or payload.get("downloadedFileBytes")
    if isinstance(raw_bytes, (bytes, bytearray)) and len(raw_bytes) <= _INLINE_FILE_BYTES_LIMIT:
        binary = bytes(raw_bytes)
        return {
            "base64": base64.b64encode(binary).decode("ascii"),
            "sha256": hashlib.sha256(binary).hexdigest(),
        }
    return None


def _extract_message_text(payload: Mapping[str, Any]) -> str:
    for value in (
        _path_value(payload, "text", "content"),
        _path_value(payload, "content", "text"),
        _path_value(payload, "content", "content"),
        payload.get("text"),
        payload.get("content"),
        _path_value(payload, "msgParam", "content"),
    ):
        if isinstance(value, Mapping):
            candidate = _first_text(value.get("content"), value.get("text"))
        else:
            candidate = _first_text(value)
        if candidate:
            return candidate
    return ""


def _stable_trace_id(payload: Mapping[str, Any]) -> str:
    return build_dingtalk_fallback_identity(payload)


def build_dingtalk_fallback_identity(payload: Mapping[str, Any]) -> str:
    direct = _first_text(
        payload.get("traceId"),
        payload.get("trace_id"),
        payload.get("messageId"),
        payload.get("msgId"),
        payload.get("message_id"),
    )
    if direct:
        return direct

    canonical_payload = {
        "conversationId": _first_text(payload.get("conversationId"), payload.get("chatId"), payload.get("openConversationId")),
        "conversationType": _first_text(payload.get("conversationType"), payload.get("chatType"), payload.get("chat_type")),
        "senderId": _first_text(payload.get("senderId"), payload.get("sender_id"), payload.get("senderUserId")),
        "senderStaffId": _first_text(payload.get("senderStaffId"), payload.get("sender_staff_id"), payload.get("senderId")),
        "senderUnionId": _first_text(payload.get("senderUnionId"), payload.get("sender_union_id"), payload.get("unionId")),
        "senderNick": _first_text(payload.get("senderNick"), payload.get("sender_nick")),
        "conversationTitle": _first_text(payload.get("conversationTitle"), payload.get("conversation_title")),
        "eventTime": _first_text(
            payload.get("createAt"),
            payload.get("create_at"),
            payload.get("eventTime"),
            payload.get("event_time"),
            payload.get("messageTime"),
            payload.get("timestamp"),
        ),
        "eventType": _first_text(
            payload.get("msgtype"),
            payload.get("messageType"),
            payload.get("message_type"),
            payload.get("eventType"),
            payload.get("event_type"),
            payload.get("type"),
        ),
        "text": _extract_message_text(payload),
        "fileName": _first_text(payload.get("fileName"), payload.get("file_name")),
        "fileId": _first_text(payload.get("fileId"), payload.get("file_id")),
        "mediaId": _first_text(payload.get("mediaId"), payload.get("media_id")),
        "rawEvent": _sanitize_raw_event(payload, excluded_keys=_IDENTITY_EXCLUDED_KEYS),
    }
    base = json.dumps(
        {key: value for key, value in canonical_payload.items() if value not in (None, "", {}, [])},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return f"dingtalk-stream-sha256:{digest}"


def _path_value(payload: Mapping[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        current = _coerce_mapping(current)
        if current is None:
            return None
        current = current.get(key)
    return current


def _coerce_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _sanitize_raw_event(value: Any, *, depth: int = 0, excluded_keys: set[str] | None = None) -> Any:
    if depth >= 6:
        return "...[truncated]"
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, item in list(value.items())[:25]:
            key_text = str(raw_key)
            lowered = key_text.lower().replace("-", "_")
            if excluded_keys and lowered in excluded_keys:
                continue
            if lowered in _INLINE_FILE_RAW_KEYS:
                continue
            if any(marker in lowered for marker in ("token", "secret", "authorization", "webhook")):
                continue
            if lowered in _SECRET_QUERY_KEYS:
                continue
            sanitized[key_text] = _sanitize_raw_event(item, depth=depth + 1, excluded_keys=excluded_keys)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        items = list(value)[:25]
        return [_sanitize_raw_event(item, depth=depth + 1, excluded_keys=excluded_keys) for item in items]
    if isinstance(value, str):
        return _sanitize_error_text(value)
    return value


def _response_json(response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _sanitize_error_text(text: str, *, secrets: Sequence[str] = ()) -> str:
    sanitized = redact_sensitive_text(str(text or ""))
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(secret, "<redacted>")
    sanitized = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=<redacted>", sanitized)

    def _redact_url(match: re.Match[str]) -> str:
        try:
            parts = urlsplit(match.group(0))
            query = parse_qsl(parts.query, keep_blank_values=True)
            if not query:
                return match.group(0)
            redacted_query = []
            changed = False
            for key, value in query:
                if key.lower() in _SECRET_QUERY_KEYS:
                    redacted_query.append((key, "<redacted>"))
                    changed = True
                else:
                    redacted_query.append((key, value))
            if not changed:
                return match.group(0)
            return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(redacted_query), parts.fragment))
        except Exception:  # noqa: BLE001
            return match.group(0)

    sanitized = re.sub(r"https?://[^\s\"'<>]+", _redact_url, sanitized)
    return sanitized


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            cleaned = str(value).strip()
            if cleaned:
                return cleaned
    return ""


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}
