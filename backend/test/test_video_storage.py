"""Unit tests for the video storage backends (local disk and GCS)."""

import pytest

from src.utils import clip_scribe_video_storage as vs_mod
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


def test_local_signed_url_is_none(tmp_path):
    # Local bytes are served via FileResponse, so there is no signed URL.
    assert LocalVideoStorage(tmp_path).signed_url("anything.mp4") is None


def test_factory_selects_local(tmp_path):
    assert isinstance(make_video_storage("local", tmp_path), LocalVideoStorage)


def test_factory_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError):
        make_video_storage("s3", tmp_path)


# ── GCS backend ──────────────────────────────────────────────────────────────
# The bucket is faked in-memory so the seam's behavior (key layout, upload,
# dedup lookup, scratch download/cleanup, signing) is exercised without network.


class _FakeBlob:
    def __init__(self, bucket: "_FakeBucket", name: str) -> None:
        self._bucket = bucket
        self.name = name

    def upload_from_filename(self, path: str) -> None:
        with open(path, "rb") as f:
            self._bucket.objects[self.name] = f.read()

    def download_to_filename(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(self._bucket.objects[self.name])

    def exists(self) -> bool:
        return self.name in self._bucket.objects

    def generate_signed_url(self, *, version, expiration, method) -> str:
        return f"https://signed.example/{self._bucket.name}/{self.name}?m={method}"


class _FakeBucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self.objects: dict[str, bytes] = {}

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self, name)


class _FakeClient:
    def __init__(self) -> None:
        self.buckets: dict[str, _FakeBucket] = {}

    def bucket(self, name: str) -> _FakeBucket:
        return self.buckets.setdefault(name, _FakeBucket(name))


@pytest.fixture
def gcs(tmp_path):
    client = _FakeClient()
    storage = GCSVideoStorage("clipscribe", tmp_path, client=client)
    return storage, client


def test_gcs_commit_uploads_under_videos_prefix_and_returns_key(gcs, tmp_path):
    storage, client = gcs
    staged = storage.staging_dir() / "stage.tmp"
    staged.write_bytes(b"payload")

    key = storage.commit("user42", staged, ".mp4")

    assert key.startswith("videos/user42/")
    assert key.endswith(".mp4")
    assert not staged.exists()  # staged copy consumed by the upload
    assert client.bucket("clipscribe").objects[key] == b"payload"
    assert storage.exists(key)


def test_gcs_exists_false_for_unknown_key(gcs):
    storage, _ = gcs
    assert not storage.exists("videos/user42/nope.mp4")


def test_gcs_materialize_downloads_scratch_and_release_deletes_it(gcs):
    storage, _ = gcs
    staged = storage.staging_dir() / "stage.tmp"
    staged.write_bytes(b"payload")
    key = storage.commit("user42", staged, ".mp4")

    local = storage.materialize(key)
    assert local.exists()
    assert local.read_bytes() == b"payload"
    assert local.suffix == ".mp4"  # suffix preserved for format sniffing

    # release deletes only the scratch copy; the bucket object is untouched.
    storage.release(local)
    assert not local.exists()
    assert storage.exists(key)


def test_gcs_signed_url_is_a_get_url(gcs):
    storage, _ = gcs
    url = storage.signed_url("videos/user42/x.mp4")
    assert url is not None
    assert "videos/user42/x.mp4" in url
    assert "m=GET" in url


def test_factory_selects_gcs(tmp_path, monkeypatch):
    monkeypatch.setattr(vs_mod, "_make_gcs_client", lambda: _FakeClient())
    storage = make_video_storage("gcs", tmp_path, "clipscribe")
    assert isinstance(storage, GCSVideoStorage)


def test_factory_gcs_requires_bucket(tmp_path):
    with pytest.raises(ValueError):
        make_video_storage("gcs", tmp_path)
