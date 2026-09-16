from __future__ import annotations

import os
import re
import shutil
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional
from uuid import uuid4

import yaml


ATOMIC_SNAPSHOT_STORAGE = "t3_versioned_atomic"
ATOMIC_SNAPSHOT_STORAGE_VERSION = 1
VERSIONS_DIRECTORY_NAME = ".versions"
VERSION_PREFIX = "snapshot-"
STAGING_PREFIX = ".staging-"
INDEX_TEMP_PREFIX = ".map_index-"
COMMITTED_INDEX_NAME = "map_index.yaml"


SnapshotWriter = Callable[[Path], Mapping[str, Path]]
SnapshotValidator = Callable[[Path], None]


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-")
    return (label or "snapshot")[:48]


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(directory: Path) -> None:
    directory = Path(directory)
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.is_symlink():
            _fsync_file(path)
    directories = [directory]
    directories.extend(
        path
        for path in directory.rglob("*")
        if path.is_dir() and not path.is_symlink()
    )
    for path in sorted(
        directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(path)


def _remove_owned_path(
    path: Path,
    *,
    parent: Path,
    required_prefix: Optional[str] = None,
) -> None:
    path = Path(path)
    parent = Path(parent)
    if path.parent != parent:
        raise ValueError(f"Refusing to remove path outside {parent}: {path}")
    if required_prefix is not None and not path.name.startswith(
        required_prefix
    ):
        raise ValueError(f"Refusing to remove unowned path: {path}")
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _read_metadata(path: Path) -> Optional[dict]:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def _referenced_version(metadata: Optional[dict]) -> Optional[str]:
    if not metadata:
        return None
    if metadata.get("snapshot_storage") != ATOMIC_SNAPSHOT_STORAGE:
        return None
    try:
        storage_version = int(metadata.get("snapshot_storage_version", -1))
    except (TypeError, ValueError):
        return None
    if storage_version != ATOMIC_SNAPSHOT_STORAGE_VERSION:
        return None
    value = str(metadata.get("snapshot_version", ""))
    if (
        not value.startswith(VERSION_PREFIX)
        or Path(value).name != value
    ):
        return None
    return value


def _relative_child(path: Path, parent: Path, *, name: str) -> Path:
    path = Path(path)
    parent = Path(parent)
    try:
        relative = path.relative_to(parent)
    except ValueError as exception:
        raise ValueError(f"{name} is outside staging directory: {path}") from exception
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"Invalid {name}: {path}")
    return relative


def _relative_payload_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Invalid snapshot payload path: {value!r}")
    return path


def _committed_metadata(
    *,
    version_metadata: dict,
    version_name: str,
    version_index_relative: Path,
    output_directory: Path,
    final_directory: Path,
) -> dict:
    metadata = deepcopy(version_metadata)
    entries = metadata.get("tiles", [])
    if not isinstance(entries, list):
        raise ValueError("Snapshot tiles metadata must be a list")
    index_parent = version_index_relative.parent
    for entry in entries:
        if not isinstance(entry, dict) or "file" not in entry:
            raise ValueError("Snapshot tile entry is missing file metadata")
        payload = _relative_payload_path(entry["file"])
        relative = (
            Path(VERSIONS_DIRECTORY_NAME)
            / version_name
            / index_parent
            / payload
        )
        target = output_directory / relative
        try:
            target.relative_to(final_directory)
        except ValueError as exception:
            raise ValueError(
                f"Snapshot payload escapes version directory: {target}"
            ) from exception
        if not target.is_file():
            raise ValueError(f"Committed snapshot payload is missing: {target}")
        entry["file"] = relative.as_posix()
    metadata["snapshot_storage"] = ATOMIC_SNAPSHOT_STORAGE
    metadata["snapshot_storage_version"] = (
        ATOMIC_SNAPSHOT_STORAGE_VERSION
    )
    metadata["snapshot_version"] = version_name
    return metadata


def _write_yaml_fsync(path: Path, metadata: dict) -> None:
    text = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _manifest_references_version(index_path: Path, version_name: str) -> bool:
    return _referenced_version(_read_metadata(index_path)) == version_name


def _clean_abandoned_work(
    output_directory: Path,
    versions_directory: Path,
) -> None:
    for path in tuple(versions_directory.iterdir()):
        if path.name.startswith(STAGING_PREFIX):
            _remove_owned_path(
                path,
                parent=versions_directory,
                required_prefix=STAGING_PREFIX,
            )
    for path in tuple(output_directory.iterdir()):
        if (
            path.name.startswith(INDEX_TEMP_PREFIX)
            and path.name.endswith(".tmp")
        ):
            _remove_owned_path(
                path,
                parent=output_directory,
                required_prefix=INDEX_TEMP_PREFIX,
            )


def _prepare_retention(
    *,
    output_directory: Path,
    versions_directory: Path,
    old_version: Optional[str],
    new_version: str,
) -> None:
    # The committed version and its immediate predecessor are retained.  Do
    # not prune on the first migration from a legacy or unreadable manifest;
    # the next successful versioned save has an unambiguous ownership set.
    if old_version is None:
        return
    keep = {old_version, new_version}
    for path in tuple(versions_directory.iterdir()):
        if (
            path.name.startswith(VERSION_PREFIX)
            and path.name not in keep
        ):
            _remove_owned_path(
                path,
                parent=versions_directory,
                required_prefix=VERSION_PREFIX,
            )

    # A pre-versioned snapshot used grid_map_snapshot/tiles directly.  Once
    # the already-committed manifest is versioned, that flat directory is no
    # longer referenced and can be removed without affecting the rollback
    # version.  The first migration deliberately leaves it for one cycle.
    legacy_tiles = output_directory / "tiles"
    if legacy_tiles.exists() or legacy_tiles.is_symlink():
        _remove_owned_path(
            legacy_tiles,
            parent=output_directory,
        )


def commit_atomic_snapshot(
    output_directory: Path,
    *,
    version_label: str,
    write_snapshot: SnapshotWriter,
    validate_snapshot: SnapshotValidator,
) -> Dict[str, Path]:
    """Commit one immutable snapshot through an atomic root manifest.

    Tile files are written and validated in a private staging directory.  A
    rename makes that version immutable, then ``map_index.yaml`` is replaced
    atomically as the sole commit point.  Until that final replacement, the
    previous root manifest and all files it references remain untouched.
    The caller must serialize writers for a given output directory.
    """
    output_directory = Path(output_directory).absolute()
    output_directory.mkdir(parents=True, exist_ok=True)
    versions_directory = output_directory / VERSIONS_DIRECTORY_NAME
    versions_directory.mkdir(parents=True, exist_ok=True)
    _clean_abandoned_work(output_directory, versions_directory)

    committed_index = output_directory / COMMITTED_INDEX_NAME
    old_metadata = _read_metadata(committed_index)
    old_version = _referenced_version(old_metadata)
    unique = f"{time.time_ns()}-{uuid4().hex[:8]}"
    version_name = f"{VERSION_PREFIX}{_safe_label(version_label)}-{unique}"
    staging_directory = versions_directory / f"{STAGING_PREFIX}{unique}"
    final_directory = versions_directory / version_name
    temporary_index = output_directory / (
        f"{INDEX_TEMP_PREFIX}{uuid4().hex}.tmp"
    )
    final_created = False

    try:
        staging_directory.mkdir()
        written = {
            str(name): Path(path)
            for name, path in write_snapshot(staging_directory).items()
        }
        if "index" not in written or "tiles" not in written:
            raise ValueError("Snapshot writer must return index and tiles paths")
        relative_paths = {
            name: _relative_child(path, staging_directory, name=name)
            for name, path in written.items()
        }
        version_index = written["index"]
        if not version_index.is_file():
            raise ValueError(f"Snapshot index was not written: {version_index}")

        _fsync_tree(staging_directory)
        validate_snapshot(version_index)
        version_metadata = _read_metadata(version_index)
        if version_metadata is None:
            raise ValueError(f"Snapshot index is unreadable: {version_index}")

        os.replace(staging_directory, final_directory)
        final_created = True
        _fsync_directory(versions_directory)
        metadata = _committed_metadata(
            version_metadata=version_metadata,
            version_name=version_name,
            version_index_relative=relative_paths["index"],
            output_directory=output_directory,
            final_directory=final_directory,
        )
        _write_yaml_fsync(temporary_index, metadata)
        # Parse the exact candidate that will become authoritative.  Payload
        # values were already checked once in staging; here only the rewritten
        # immutable paths and YAML commit object need verification.
        candidate = _read_metadata(temporary_index)
        if candidate is None or _referenced_version(candidate) != version_name:
            raise ValueError("Atomic snapshot commit manifest is invalid")
        for entry in candidate.get("tiles", []):
            if not (output_directory / str(entry["file"])).is_file():
                raise ValueError(
                    f"Atomic snapshot manifest references a missing tile: {entry}"
                )

        _fsync_directory(output_directory)
        os.replace(temporary_index, committed_index)
        _fsync_directory(output_directory)

        # Retention is deliberately post-commit.  A failed manifest switch
        # must not reduce the previous current+rollback version set.  Cleanup
        # failure after a successful commit is non-fatal: both versions
        # referenced by the policy remain complete, and a later save retries
        # pruning obsolete project-owned versions.
        try:
            _prepare_retention(
                output_directory=output_directory,
                versions_directory=versions_directory,
                old_version=old_version,
                new_version=version_name,
            )
            _fsync_directory(versions_directory)
            _fsync_directory(output_directory)
        except OSError as exception:
            warnings.warn(
                "Snapshot committed, but obsolete-version cleanup will be "
                f"retried later: {exception}",
                RuntimeWarning,
                stacklevel=2,
            )

        result = {
            name: final_directory / relative
            for name, relative in relative_paths.items()
        }
        result["index"] = committed_index
        return result
    except BaseException:
        if temporary_index.exists() or temporary_index.is_symlink():
            _remove_owned_path(
                temporary_index,
                parent=output_directory,
                required_prefix=INDEX_TEMP_PREFIX,
            )
        if staging_directory.exists() or staging_directory.is_symlink():
            _remove_owned_path(
                staging_directory,
                parent=versions_directory,
                required_prefix=STAGING_PREFIX,
            )
        if (
            final_created
            and final_directory.exists()
            and not _manifest_references_version(
                committed_index,
                version_name,
            )
        ):
            _remove_owned_path(
                final_directory,
                parent=versions_directory,
                required_prefix=VERSION_PREFIX,
            )
        raise
