from __future__ import annotations

from pathlib import Path

import pytest

from ops.file_lock import FileLockBusyError, FileLockError, file_lock


def test_file_lock_is_private_persistent_and_detects_contention(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.lock"

    with file_lock(path):
        assert path.is_file()
        assert path.stat().st_mode & 0o777 == 0o600
        with pytest.raises(FileLockBusyError):
            with file_lock(path, blocking=False):
                pass

    with file_lock(path, blocking=False):
        pass


def test_file_lock_rejects_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "outside"
    target.write_text("do not touch", encoding="utf-8")
    path = tmp_path / "state.lock"
    path.symlink_to(target)

    with pytest.raises(FileLockError):
        with file_lock(path):
            pass

    assert target.read_text(encoding="utf-8") == "do not touch"


def test_file_lock_preserves_protected_body_exception(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.lock"

    with pytest.raises(OSError, match="transaction failed"):
        with file_lock(path):
            raise OSError("transaction failed")
