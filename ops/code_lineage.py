"""Content-bound lineage for a repo-local Python import closure.

Research artifacts must become stale when any local implementation they can
execute changes, including transitive and function-local imports.  This module
walks the static Python import graph rooted at one entry point and binds every
repo-local source file by path, size, and SHA-256.  Filesystem timestamps are
deliberately excluded from the identity.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import stat as stat_module
from pathlib import Path

from ops.artifact_provenance import (
    ArtifactSourceChangedError,
    file_set_identity,
)


LINEAGE_SCHEMA = "repo_local_python_code_lineage_v1"


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _regular_source(path: Path) -> Path | None:
    """Return an existing Python source candidate without following symlinks."""
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return None
    if not stat_module.S_ISREG(stat.st_mode):
        raise ArtifactSourceChangedError(
            f"local Python dependency is not a regular file: {path}"
        )
    if stat.st_nlink != 1:
        raise ArtifactSourceChangedError(
            f"local Python dependency has multiple hard links: {path}"
        )
    return path


def _module_sources(parts: tuple[str, ...], root: Path) -> set[Path]:
    """Resolve one absolute module name to local module/package sources."""
    if not parts or any(not part.isidentifier() for part in parts):
        return set()
    sources: set[Path] = set()
    for index in range(1, len(parts)):
        candidate = root.joinpath(*parts[:index], "__init__.py")
        source = _regular_source(candidate)
        if source is not None:
            sources.add(source)
    module_source = _regular_source(
        root.joinpath(*parts).with_suffix(".py")
    )
    if module_source is not None:
        sources.add(module_source)
    package_source = _regular_source(
        root.joinpath(*parts, "__init__.py")
    )
    if package_source is not None:
        sources.add(package_source)
    return sources


def _import_sources(source: Path, root: Path) -> set[Path]:
    """Return every syntactically imported repo-local source from one file."""
    try:
        tree = ast.parse(source.read_bytes(), filename=str(source))
    except (OSError, SyntaxError) as exc:
        raise ArtifactSourceChangedError(
            f"local Python dependency could not be parsed: {source}"
        ) from exc

    imported: set[Path] = set()
    package = source.relative_to(root).parent.parts
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.update(
                    _module_sources(tuple(alias.name.split(".")), root)
                )
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            parent_count = node.level - 1
            if parent_count > len(package):
                continue
            base = package[: len(package) - parent_count]
        else:
            base = ()
        module = tuple(node.module.split(".")) if node.module else ()
        module_parts = (*base, *module)
        imported.update(_module_sources(module_parts, root))
        for alias in node.names:
            if alias.name != "*":
                imported.update(
                    _module_sources(
                        (*module_parts, *alias.name.split(".")),
                        root,
                    )
                )
    return imported


def _dependency_paths(entrypoint: Path, root: Path) -> dict[str, Path]:
    entrypoint = _absolute(entrypoint)
    root = _absolute(root)
    try:
        entrypoint.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"entrypoint is outside repository root: {entrypoint}"
        ) from exc
    if _regular_source(entrypoint) is None:
        raise FileNotFoundError(entrypoint)

    pending = [entrypoint]
    visited: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        pending.extend(
            sorted(
                _import_sources(source, root) - visited,
                key=lambda path: path.as_posix(),
                reverse=True,
            )
        )
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(
            visited,
            key=lambda value: value.relative_to(root).as_posix(),
        )
    }


def python_code_lineage(*, entrypoint: Path, root: Path) -> dict:
    """Capture one coherent, content-only repo-local Python import closure."""
    root = _absolute(root)
    paths_before = _dependency_paths(entrypoint, root)
    identities_before = file_set_identity(paths_before, root=root)
    paths_after = _dependency_paths(entrypoint, root)
    if paths_before != paths_after:
        raise ArtifactSourceChangedError(
            "local Python import graph changed during lineage capture"
        )
    identities_after = file_set_identity(paths_after, root=root)
    if identities_before != identities_after:
        raise ArtifactSourceChangedError(
            "local Python source changed during lineage capture"
        )
    body = {
        "schema": LINEAGE_SCHEMA,
        "entrypoint": _absolute(entrypoint).relative_to(root).as_posix(),
        "files": identities_before,
    }
    return {
        **body,
        "lineage_sha256": hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
