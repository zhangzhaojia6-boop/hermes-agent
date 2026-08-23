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
