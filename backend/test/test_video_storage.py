"""Unit tests for the local video storage backend (src/utils/video_storage.py)."""

import pytest

from src.utils.clip_scribe_video_storage import (
    GCSVideoStorage,
    LocalVideoStorage,
    make_video_storage,
)


def test_commit_moves_staged_file_and_returns_key(tmp_path):
    storage = LocalVideoStorage(tmp_path)
    staged = storage.staging_dir() / "stage.tmp"
    staged.write_bytes(b"payload")

    key = storage.commit("local", staged, ".mp4")

    assert key.endswith(".mp4")
    assert not staged.exists()  # consumed by the move
    assert (tmp_path / key).read_bytes() == b"payload"
    assert storage.exists(key)


def test_materialize_returns_stored_path_and_release_is_noop(tmp_path):
    storage = LocalVideoStorage(tmp_path)
    staged = storage.staging_dir() / "stage.tmp"
    staged.write_bytes(b"payload")
    key = storage.commit("local", staged, ".mp4")

    local = storage.materialize(key)
    assert local == tmp_path / key

    # release must NOT delete the source: the materialized path IS the object.
    storage.release(local)
    assert local.exists()


def test_exists_false_for_unknown_key(tmp_path):
    assert not LocalVideoStorage(tmp_path).exists("nope.mp4")


def test_factory_selects_local(tmp_path):
    assert isinstance(make_video_storage("local", tmp_path), LocalVideoStorage)


def test_gcs_backend_fails_fast(tmp_path):
    # Reserved backend: constructing it raises so a misconfig is caught at startup.
    with pytest.raises(NotImplementedError):
        make_video_storage("gcs", tmp_path)
    with pytest.raises(NotImplementedError):
        GCSVideoStorage()


def test_factory_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError):
        make_video_storage("s3", tmp_path)
