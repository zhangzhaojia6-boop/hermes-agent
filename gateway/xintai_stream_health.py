from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class XintaiStreamHealth:
    connected: bool = False
    stream_running: bool = False
    last_event_at: datetime | None = None
    relay_success_count: int = 0
    relay_failure_count: int = 0
    last_error: str | None = None

    def set_stream_running(self, running: bool) -> None:
        active = bool(running)
        self.stream_running = active
        self.connected = active

    def record_event(self, when: datetime | None = None) -> None:
        self.last_event_at = when or datetime.now(timezone.utc)

    def record_relay(self, *, accepted: bool, error: str | None = None, category: str | None = None) -> None:
        if category == "disabled":
            return
        if accepted:
            self.relay_success_count += 1
            self.last_error = None
            return
        self.relay_failure_count += 1
        self.last_error = error

    def snapshot(self) -> dict[str, object]:
        return {
            "connected": self.connected,
            "stream_running": self.stream_running,
            "last_event_at": self.last_event_at,
            "relay_success_count": self.relay_success_count,
            "relay_failure_count": self.relay_failure_count,
            "last_error": self.last_error,
        }
