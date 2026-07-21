from tests.gateway import conftest as gateway_conftest


def test_gateway_test_inventory_ignores_pycache(tmp_path, monkeypatch):
    source_dir = tmp_path / "platforms"
    source_dir.mkdir()
    source = source_dir / "test_adapter.py"
    source.write_text("", encoding="utf-8")

    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "test_stale.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(gateway_conftest, "_GATEWAY_DIR", tmp_path)

    assert gateway_conftest._iter_gateway_test_paths() == [source]
