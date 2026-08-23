from types import SimpleNamespace

from agent.conversation_loop import _force_non_streaming_for_endpoint


def test_openrouter_can_be_forced_to_non_streaming_without_affecting_fallback(monkeypatch):
    monkeypatch.setenv("HERMES_OPENROUTER_FORCE_NON_STREAMING", "1")

    assert _force_non_streaming_for_endpoint(
        SimpleNamespace(base_url="https://openrouter.ai/api/v1")
    )
    assert not _force_non_streaming_for_endpoint(
        SimpleNamespace(base_url="https://chatgpt.com/backend-api/codex")
    )


def test_configured_openrouter_bridge_can_be_forced_to_non_streaming(monkeypatch):
    monkeypatch.setenv("HERMES_OPENROUTER_FORCE_NON_STREAMING", "1")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://127.0.0.1:7898/v1/")

    assert _force_non_streaming_for_endpoint(
        SimpleNamespace(base_url="http://127.0.0.1:7898/v1")
    )
