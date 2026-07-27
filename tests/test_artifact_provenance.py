from __future__ import annotations

import os
from pathlib import Path

import pytest

import ops.artifact_provenance as provenance


def test_file_identity_rejects_mutation_during_hash(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.bin"
    source.write_bytes(b"before")
    original = provenance._hash_file_handle

    def mutate_while_open(handle):
        digest = original(handle)
        # Same-size rewrite proves this does not rely on size alone.
        source.write_bytes(b"after!")
        return digest

    monkeypatch.setattr(
        provenance,
        "_hash_file_handle",
        mutate_while_open,
    )

    with pytest.raises(
        provenance.ArtifactSourceChangedError,
        match="changed while hashing",
    ):
        provenance.file_identity(source, root=tmp_path)


def test_file_set_identity_rejects_sibling_mutation_between_hashes(
    tmp_path,
    monkeypatch,
):
    main = tmp_path / "candles.db"
    wal = tmp_path / "candles.db-wal"
    main.write_bytes(b"main")
    wal.write_bytes(b"wal-before")
    original = provenance._hash_file_handle
    calls = 0

    def mutate_wal_after_main(handle):
        nonlocal calls
        digest = original(handle)
        calls += 1
        if calls == 1:
            wal.write_bytes(b"wal-after")
        return digest

    monkeypatch.setattr(
        provenance,
        "_hash_file_handle",
        mutate_wal_after_main,
    )

    with pytest.raises(
        provenance.ArtifactSourceChangedError,
        match="source set changed",
    ):
        provenance.file_set_identity(
            {"main": main, "wal": wal},
            root=tmp_path,
        )


def test_tree_identity_rejects_membership_change_during_hash(
    tmp_path,
    monkeypatch,
):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "first.json").write_text("{}", encoding="utf-8")
    original = provenance._hash_file_handle
    mutated = False

    def add_tree_member(handle):
        nonlocal mutated
        digest = original(handle)
        if not mutated:
            mutated = True
            (tree / "second.json").write_text("{}", encoding="utf-8")
        return digest

    monkeypatch.setattr(
        provenance,
        "_hash_file_handle",
        add_tree_member,
    )

    with pytest.raises(
        provenance.ArtifactSourceChangedError,
        match="source tree",
    ):
        provenance.tree_identity(
            tree,
            root=tmp_path,
            suffixes={".json"},
        )


def test_file_identity_rejects_symlink_source(tmp_path):
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    source = tmp_path / "source.bin"
    source.symlink_to(target)

    with pytest.raises(
        provenance.ArtifactSourceChangedError,
        match="not a regular file",
    ):
        provenance.file_identity(source, root=tmp_path)


def test_file_identity_rejects_symlink_swap_before_open(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.bin"
    source.write_bytes(b"inside")
    referent = tmp_path / "outside.bin"
    referent.write_bytes(b"outside")
    original_open = provenance.os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if Path(path) == source and not swapped:
            swapped = True
            source.unlink()
            source.symlink_to(referent)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(provenance.os, "open", swap_before_open)

    with pytest.raises(
        provenance.ArtifactSourceChangedError,
        match="opened safely",
    ):
        provenance.file_identity(source, root=tmp_path)


def test_tree_identity_rejects_symlink_member(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    (tree / "linked.json").symlink_to(target)

    with pytest.raises(
        provenance.ArtifactSourceChangedError,
        match="contains a symlink",
    ):
        provenance.tree_identity(
            tree,
            root=tmp_path,
            suffixes={".json"},
        )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"safe": 1, "safe": 2}',
        b'{"safe": NaN}',
        b'{"safe": Infinity}',
    ],
)
def test_strict_json_object_bytes_rejects_ambiguous_json(raw):
    with pytest.raises(provenance.ArtifactValidationError):
        provenance.strict_json_object_bytes(raw, source="test")


def test_strict_json_object_rejects_mutation_during_read(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "state.json"
    source.write_text('{"state":"before"}', encoding="utf-8")
    original = provenance.Path.read_bytes

    def mutate_after_read(path):
        content = original(path)
        if path == source:
            source.write_text('{"state":"after!"}', encoding="utf-8")
        return content

    monkeypatch.setattr(provenance.Path, "read_bytes", mutate_after_read)

    with pytest.raises(
        provenance.ArtifactValidationError,
        match="changed while reading",
    ):
        provenance.strict_json_object(source)


def test_strict_json_object_rejects_symlink(tmp_path):
    target = tmp_path / "target.json"
    target.write_text('{"state":"outside"}', encoding="utf-8")
    source = tmp_path / "state.json"
    source.symlink_to(target)

    with pytest.raises(
        provenance.ArtifactValidationError,
        match="invalid JSON artifact",
    ):
        provenance.strict_json_object(source)


def test_atomic_write_rejects_symlink_without_touching_referent(tmp_path):
    referent = tmp_path / "outside.json"
    referent.write_bytes(b"do not touch")
    target = tmp_path / "artifact.json"
    target.symlink_to(referent)

    with pytest.raises(OSError, match="regular file"):
        provenance.atomic_write_bytes(target, b"replacement")

    assert referent.read_bytes() == b"do not touch"


def test_atomic_write_rejects_symlinked_parent(tmp_path):
    actual_parent = tmp_path / "outside"
    actual_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(OSError, match="real directory"):
        provenance.atomic_write_bytes(
            linked_parent / "artifact.json",
            b"replacement",
        )

    assert list(actual_parent.iterdir()) == []


def test_atomic_write_rejects_target_appearing_during_write(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "artifact.json"
    original_fsync = provenance.os.fsync
    injected = False

    def inject_target(fd):
        nonlocal injected
        original_fsync(fd)
        if not injected:
            injected = True
            target.write_bytes(b"concurrent writer")

    monkeypatch.setattr(provenance.os, "fsync", inject_target)

    with pytest.raises(OSError, match="appeared during write"):
        provenance.atomic_write_bytes(target, b"ours")

    assert target.read_bytes() == b"concurrent writer"
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_write_new_artifact_is_private_with_permissive_umask(
    tmp_path,
):
    target = tmp_path / "artifact.json"
    previous_umask = os.umask(0)
    try:
        provenance.atomic_write_bytes(target, b"private")
    finally:
        os.umask(previous_umask)

    assert target.stat().st_mode & 0o777 == 0o600


def test_atomic_write_existing_public_artifact_keeps_temp_private(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "artifact.json"
    target.write_bytes(b"old")
    target.chmod(0o644)
    observed_modes = []
    original_fsync = provenance.os.fsync

    def inspect_temp_before_sync(fd):
        temporary = list(tmp_path.glob(".*.tmp"))
        if temporary:
            observed_modes.append(temporary[0].stat().st_mode & 0o777)
        original_fsync(fd)

    monkeypatch.setattr(provenance.os, "fsync", inspect_temp_before_sync)
    provenance.atomic_write_bytes(target, b"replacement")

    assert observed_modes[0] == 0o600
    assert target.stat().st_mode & 0o777 == 0o644


def test_atomic_write_rejects_target_appearing_at_publish(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "artifact.json"
    original_link = provenance.os.link

    def inject_target_then_link(source, destination, **kwargs):
        target.write_bytes(b"concurrent writer")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(provenance.os, "link", inject_target_then_link)

    with pytest.raises(OSError, match="appeared during publish"):
        provenance.atomic_write_bytes(target, b"ours")

    assert target.read_bytes() == b"concurrent writer"
    assert not list(tmp_path.glob(".*.tmp"))
