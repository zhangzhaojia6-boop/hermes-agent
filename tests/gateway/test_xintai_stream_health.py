from datetime import datetime, timedelta, timezone

from gateway.xintai_stream_health import XintaiStreamHealth


def test_stream_health_tracks_event_and_relay_counters() -> None:
    health = XintaiStreamHealth()
    earlier = datetime.now(timezone.utc) - timedelta(minutes=5)

    health.connected = True
    health.stream_running = True
    health.record_event(earlier)
    health.record_relay(accepted=True)
    health.record_relay(accepted=False, error="relay_http_error")

    snapshot = health.snapshot()
    assert snapshot["connected"] is True
    assert snapshot["stream_running"] is True
    assert snapshot["last_event_at"] == earlier
    assert snapshot["relay_success_count"] == 1
    assert snapshot["relay_failure_count"] == 1
    assert snapshot["last_error"] == "relay_http_error"


def test_stream_health_ignores_disabled_relay_state() -> None:
    health = XintaiStreamHealth()

    health.record_relay(accepted=False, error="should-not-stick", category="disabled")

    snapshot = health.snapshot()
    assert snapshot["stream_running"] is False
    assert snapshot["relay_success_count"] == 0
    assert snapshot["relay_failure_count"] == 0
    assert snapshot["last_error"] is None
