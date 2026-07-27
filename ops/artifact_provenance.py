"""Small, strict primitives for content-bound derived artifacts.

The helpers in this module deliberately avoid timestamps and mtimes in source
identities.  An artifact is current only when the exact source bytes and the
generator contract that produced it still match.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat as stat_module
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping


class ArtifactValidationError(ValueError):
    """A derived artifact is malformed, tampered with, or no longer current."""


class ArtifactSourceChangedError(RuntimeError):
    """A source could not be identified from one stable filesystem version."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON representation used for all digests."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file_handle(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _stat_token(value: os.stat_result) -> tuple[int, ...]:
    """Return metadata used only to prove one stable read, not as identity."""
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _regular_file_token(path: Path) -> tuple[int, ...] | None:
    try:
        # Provenance paths must identify the named file itself. Following a
        # symlink would let an otherwise self-contained manifest silently bind
        # to mutable bytes outside the declared tree.
        value = path.lstat()
    except FileNotFoundError:
        return None
    if not stat_module.S_ISREG(value.st_mode):
        raise ArtifactSourceChangedError(
            f"artifact source is not a regular file: {path}"
        )
    return _stat_token(value)


def _directory_token(path: Path) -> tuple[int, ...] | None:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    if not stat_module.S_ISDIR(value.st_mode):
        raise ArtifactSourceChangedError(
            f"artifact source tree is not a directory: {path}"
        )
    return _stat_token(value)


def _stable_file_details(path: Path) -> tuple[int, str] | None:
    """Hash one open inode and reject replacement or mutation during the read."""
    path_before = _regular_file_token(path)
    if path_before is None:
        # A second observation closes the common absent->present race. A
        # caller that needs a stable group also uses ``file_set_identity``.
        if _regular_file_token(path) is not None:
            raise ArtifactSourceChangedError(
                f"artifact source appeared during identity capture: {path}"
            )
        return None

    if not hasattr(os, "O_NOFOLLOW"):
        raise ArtifactSourceChangedError(
            "O_NOFOLLOW is required for safe artifact source reads"
        )
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as handle:
            fd_before = _stat_token(os.fstat(handle.fileno()))
            if fd_before != path_before:
                raise ArtifactSourceChangedError(
                    f"artifact source was replaced before hashing: {path}"
                )
            digest = _hash_file_handle(handle)
            # Filesystems may expose coarse or unchanged timestamp metadata
            # for a same-size rewrite. A second pass over the same open inode
            # prevents a mixed first-pass digest from being accepted solely
            # because fstat metadata happened to remain equal.
            handle.seek(0)
            verification_digest = _hash_file_handle(handle)
            fd_after = _stat_token(os.fstat(handle.fileno()))
    except FileNotFoundError as exc:
        raise ArtifactSourceChangedError(
            f"artifact source disappeared before hashing: {path}"
        ) from exc
    except OSError as exc:
        raise ArtifactSourceChangedError(
            f"artifact source could not be opened safely: {path}"
        ) from exc

    path_after = _regular_file_token(path)
    if (
        digest != verification_digest
        or fd_before != fd_after
        or path_after != fd_after
        or path_before != fd_before
    ):
        raise ArtifactSourceChangedError(
            f"artifact source changed while hashing: {path}"
        )
    return int(fd_after[4]), digest


def sha256_file(path: str | Path) -> str:
    details = _stable_file_details(Path(path))
    if details is None:
        raise FileNotFoundError(f"regular file missing: {path}")
    return details[1]


def _display_path(path: Path, *, root: Path) -> str:
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(
            Path(os.path.abspath(root))
        ).as_posix()
    except ValueError:
        return str(absolute)


def resolve_identity_path(path_value: str, *, root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def file_identity(path: str | Path, *, root: Path) -> dict[str, Any]:
    source = Path(path)
    details = _stable_file_details(source)
    identity: dict[str, Any] = {
        "path": _display_path(source, root=root),
        "exists": details is not None,
    }
    if details is None:
        return identity
    size, digest = details
    identity.update(
        {
            "size": size,
            "sha256": digest,
        }
    )
    return identity


def file_set_identity(
    paths: Mapping[str, str | Path],
    *,
    root: Path,
) -> dict[str, dict[str, Any]]:
    """Capture a coherent set of files or reject any one-way group mutation."""
    sources = {name: Path(path) for name, path in paths.items()}
    before = {
        name: _regular_file_token(path)
        for name, path in sources.items()
    }
    identities = {
        name: file_identity(path, root=root)
        for name, path in sources.items()
    }
    after = {
        name: _regular_file_token(path)
        for name, path in sources.items()
    }
    if before != after:
        changed = sorted(
            name for name in sources if before[name] != after[name]
        )
        raise ArtifactSourceChangedError(
            f"artifact source set changed during identity capture: {changed}"
        )
    return identities


def _tree_files(
    source: Path,
    *,
    allowed: set[str] | None,
) -> dict[str, Path]:
    try:
        candidates = []
        for item in source.rglob("*"):
            item_stat = item.lstat()
            if stat_module.S_ISLNK(item_stat.st_mode):
                raise ArtifactSourceChangedError(
                    f"artifact source tree contains a symlink: {item}"
                )
            if (
                stat_module.S_ISREG(item_stat.st_mode)
                and (allowed is None or item.suffix in allowed)
            ):
                candidates.append(item)
    except OSError as exc:
        raise ArtifactSourceChangedError(
            f"artifact source tree changed while enumerating: {source}"
        ) from exc
    return {
        item.relative_to(source).as_posix(): item
        for item in sorted(
            candidates,
            key=lambda item: item.relative_to(source).as_posix(),
        )
    }


def source_bundle_identity(
    *,
    files: Mapping[str, str | Path],
    trees: Mapping[
        str,
        tuple[str | Path, Iterable[str] | None],
    ],
    root: Path,
) -> dict[str, dict[str, Any]]:
    """Capture files and recursive trees as one coherent source bundle."""
    direct_sources = {
        name: Path(path) for name, path in files.items()
    }
    tree_specs = {
        name: (
            Path(path),
            set(suffixes) if suffixes is not None else None,
        )
        for name, (path, suffixes) in trees.items()
    }
    directory_before = {
        name: _directory_token(path)
        for name, (path, _allowed) in tree_specs.items()
    }
    candidates_before = {
        name: (
            _tree_files(path, allowed=allowed)
            if directory_before[name] is not None
            else {}
        )
        for name, (path, allowed) in tree_specs.items()
    }

    combined: dict[str, Path] = {}
    direct_keys: dict[str, str] = {}
    for index, (name, path) in enumerate(direct_sources.items()):
        key = f"direct:{index}"
        direct_keys[name] = key
        combined[key] = path
    tree_keys: dict[str, dict[str, str]] = {}
    for tree_index, (name, candidates) in enumerate(
        candidates_before.items()
    ):
        tree_keys[name] = {}
        for file_index, (relative, path) in enumerate(candidates.items()):
            key = f"tree:{tree_index}:{file_index}"
            tree_keys[name][relative] = key
            combined[key] = path

    combined_identities = file_set_identity(combined, root=root)
    directory_after = {
        name: _directory_token(path)
        for name, (path, _allowed) in tree_specs.items()
    }
    candidates_after = {
        name: (
            _tree_files(path, allowed=allowed)
            if directory_after[name] is not None
            else {}
        )
        for name, (path, allowed) in tree_specs.items()
    }
    changed_trees = sorted(
        name
        for name in tree_specs
        if (
            directory_before[name] != directory_after[name]
            or list(candidates_before[name])
            != list(candidates_after[name])
        )
    )
    if changed_trees:
        raise ArtifactSourceChangedError(
            "artifact source trees changed during identity capture: "
            f"{changed_trees}"
        )

    file_identities = {
        name: combined_identities[key]
        for name, key in direct_keys.items()
    }
    tree_identities: dict[str, dict[str, Any]] = {}
    for name, (path, _allowed) in tree_specs.items():
        entries = []
        for relative, key in tree_keys[name].items():
            member_identity = combined_identities[key]
            if not member_identity["exists"]:
                raise ArtifactSourceChangedError(
                    f"artifact source tree member disappeared: {relative}"
                )
            entries.append(
                {
                    "path": relative,
                    "size": member_identity["size"],
                    "sha256": member_identity["sha256"],
                }
            )
        tree_identity_value: dict[str, Any] = {
            "path": _display_path(path, root=root),
            "exists": directory_before[name] is not None,
            "files": entries,
        }
        tree_identity_value["tree_sha256"] = sha256_bytes(
            canonical_json_bytes(entries)
        )
        tree_identities[name] = tree_identity_value
    return {
        "files": file_identities,
        "trees": tree_identities,
    }


def tree_identity(
    path: str | Path,
    *,
    root: Path,
    suffixes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Hash every regular file below ``path`` in deterministic path order."""
    return source_bundle_identity(
        files={},
        trees={"tree": (path, suffixes)},
        root=root,
    )["trees"]["tree"]


def with_manifest_digest(
    manifest: dict[str, Any],
    *,
    digest_key: str = "manifest_sha256",
) -> dict[str, Any]:
    if digest_key in manifest:
        raise ValueError(f"manifest already contains {digest_key}")
    result = dict(manifest)
    result[digest_key] = sha256_bytes(canonical_json_bytes(manifest))
    return result


def manifest_digest_matches(
    manifest: dict[str, Any],
    *,
    digest_key: str = "manifest_sha256",
) -> bool:
    recorded = manifest.get(digest_key)
    if not isinstance(recorded, str):
        return False
    body = {key: value for key, value in manifest.items() if key != digest_key}
    try:
        return recorded == sha256_bytes(canonical_json_bytes(body))
    except (TypeError, ValueError):
        return False


def payload_digest(
    payload: dict[str, Any],
    *,
    digest_key: str = "payload_sha256",
) -> str:
    body = {key: value for key, value in payload.items() if key != digest_key}
    return sha256_bytes(canonical_json_bytes(body))


def strict_json_object_bytes(
    content: bytes,
    *,
    source: str | Path = "<bytes>",
) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactValidationError(
                    f"duplicate JSON object key: {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ArtifactValidationError(
                    f"non-standard JSON numeric constant: {value}"
                )
            ),
        )
    except ArtifactValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            f"invalid JSON artifact: {source}"
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(
            "JSON artifact must be a JSON object"
        )
    return payload


def strict_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        before = file_identity(source, root=source.parent)
        content = source.read_bytes()
        after = file_identity(source, root=source.parent)
    except (OSError, ArtifactSourceChangedError) as exc:
        raise ArtifactValidationError(f"invalid JSON artifact: {path}") from exc
    if (
        not before["exists"]
        or before != after
        or before.get("sha256") != sha256_bytes(content)
    ):
        raise ArtifactValidationError(
            f"JSON artifact changed while reading: {path}"
        )
    return strict_json_object_bytes(content, source=path)


def fsync_directory(path: str | Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(Path(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    """Durably replace one regular artifact without following symlinks."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(os, "O_NOFOLLOW"):
        raise OSError("O_NOFOLLOW is required for safe artifact writes")

    directory_before = destination.parent.lstat()
    if not stat_module.S_ISDIR(directory_before.st_mode):
        raise OSError(
            f"artifact parent must be a real directory: {destination.parent}"
        )
    directory_flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    directory_fd = os.open(destination.parent, directory_flags)
    temp_name = (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    fd = -1
    temp_created = False
    replaced = False
    try:
        directory_opened = os.fstat(directory_fd)
        if (
            int(directory_before.st_dev),
            int(directory_before.st_ino),
        ) != (
            int(directory_opened.st_dev),
            int(directory_opened.st_ino),
        ):
            raise OSError(
                f"artifact parent changed while opening: {destination.parent}"
            )

        try:
            original = os.stat(
                destination.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            original = None
        if original is not None and not stat_module.S_ISREG(original.st_mode):
            raise OSError(
                f"artifact target must be a regular file: {destination}"
            )

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        # Do not let a permissive inherited umask expose a newly-created
        # artifact while it still has a discoverable temporary pathname.
        # Existing targets retain their explicit mode below.
        fd = os.open(temp_name, flags, 0o600, dir_fd=directory_fd)
        temp_created = True
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            if original is not None:
                # Expose the preserved destination mode only after complete
                # content is durable; a public existing artifact must not
                # make its in-progress temporary generation public.
                os.fchmod(handle.fileno(), original.st_mode & 0o7777)
                os.fsync(handle.fileno())

        try:
            current = os.stat(
                destination.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            current = None
        if original is None:
            if current is not None:
                raise OSError(
                    f"artifact target appeared during write: {destination}"
                )
        elif current is None or _stat_token(current) != _stat_token(original):
            raise OSError(
                f"artifact target changed during write: {destination}"
            )
        elif not stat_module.S_ISREG(current.st_mode):
            raise OSError(
                f"artifact target became unsafe: {destination}"
            )

        if original is None:
            try:
                # Unlike replace(), hard-link publication cannot overwrite a
                # target that appears after the final observation.
                os.link(
                    temp_name,
                    destination.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise OSError(
                    f"artifact target appeared during publish: {destination}"
                ) from exc
            os.unlink(temp_name, dir_fd=directory_fd)
        else:
            os.replace(
                temp_name,
                destination.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
        replaced = True
        os.fsync(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if temp_created and not replaced:
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    pretty = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, pretty)
