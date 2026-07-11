from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

from hermes_constants import get_hermes_home


XINTAI_SOUL_VERSION = "2026-07-11-v1"
XINTAI_SOUL_SYNC_ENV = "XINTAI_SOUL_SYNC_ENABLED"
XINTAI_RUNTIME_NAME = "鑫泰铝业智能大脑"


@dataclass(frozen=True, slots=True)
class XintaiSoulSyncResult:
    installed: bool
    reason: str
    target_path: Path
    backup_path: Path | None = None


def sync_xintai_runtime_soul() -> XintaiSoulSyncResult:
    target_path = get_hermes_home() / "SOUL.md"

    if not _env_enabled(XINTAI_SOUL_SYNC_ENV, True):
        return XintaiSoulSyncResult(installed=False, reason="disabled", target_path=target_path, backup_path=None)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        return XintaiSoulSyncResult(
            installed=False,
            reason="existing_preserved",
            target_path=target_path,
            backup_path=None,
        )

    canonical_text = _canonical_xintai_soul_path().read_text(encoding="utf-8")
    _atomic_write_text(target_path, canonical_text)
    return XintaiSoulSyncResult(
        installed=True,
        reason="installed_missing",
        target_path=target_path,
        backup_path=None,
    )


def _canonical_xintai_soul_path() -> Path:
    return Path(__file__).resolve().parents[1] / "docker" / "SOUL.md"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _env_enabled(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}
