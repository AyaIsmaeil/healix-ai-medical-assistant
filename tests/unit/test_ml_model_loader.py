import hashlib
import json
import shutil

import pytest

from ml.model_loader import BUNDLE_DIR, MLModelError, _reset_cache_for_tests, load_bundle


@pytest.fixture(autouse=True)
def _clear_cache():
    _reset_cache_for_tests()
    yield
    _reset_cache_for_tests()


def test_load_bundle_loads_the_real_vendored_bundle():
    bundle = load_bundle()

    assert bundle.metadata["n_features"] == 131
    assert bundle.metadata["n_classes"] == 41
    assert len(bundle.feature_order) == 131
    assert bundle.model.n_features_in_ == 131
    assert len(bundle.label_encoder.classes_) == 41


def test_load_bundle_caches_after_first_call():
    first = load_bundle()
    second = load_bundle()

    assert first is second


def test_load_bundle_feature_order_matches_feature_schema_file():
    schema = json.loads((BUNDLE_DIR / "feature_schema.json").read_text(encoding="utf-8"))
    bundle = load_bundle()

    assert list(bundle.feature_order) == schema["layers"]["model_feature_order"]


# --- integrity failures -----------------------------------------------------------


def _copy_bundle_to(tmp_path):
    dest = tmp_path / "bundle"
    shutil.copytree(BUNDLE_DIR, dest)
    return dest


def test_load_bundle_raises_on_checksum_mismatch(tmp_path):
    dest = _copy_bundle_to(tmp_path)
    checksums_path = dest / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["metadata.json"] = "0" * 64  # deliberately wrong
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")

    with pytest.raises(MLModelError, match="checksum mismatch"):
        load_bundle(directory=dest)


def test_load_bundle_raises_when_a_bundle_file_is_missing(tmp_path):
    dest = _copy_bundle_to(tmp_path)
    (dest / "metadata.json").unlink()

    with pytest.raises(MLModelError, match="missing metadata.json"):
        load_bundle(directory=dest)


def test_load_bundle_raises_when_checksums_json_itself_is_missing(tmp_path):
    dest = _copy_bundle_to(tmp_path)
    (dest / "checksums.json").unlink()

    with pytest.raises(MLModelError, match="missing checksums.json"):
        load_bundle(directory=dest)


def test_load_bundle_raises_when_metadata_n_features_disagrees_with_model(tmp_path):
    dest = _copy_bundle_to(tmp_path)
    metadata_path = dest / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["n_features"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    # Re-sign checksums.json so this test isolates the n_features
    # consistency check, not an incidental checksum failure from editing
    # metadata.json above.
    checksums_path = dest / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["metadata.json"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    checksums_path.write_text(json.dumps(checksums), encoding="utf-8")

    with pytest.raises(MLModelError, match="n_features"):
        load_bundle(directory=dest)


def test_load_bundle_never_caches_a_failed_load(tmp_path):
    dest = _copy_bundle_to(tmp_path)
    (dest / "metadata.json").unlink()

    with pytest.raises(MLModelError):
        load_bundle(directory=dest)

    # The real bundle must still load cleanly afterward — a failed
    # load against a broken temp directory must not have poisoned the
    # module-level cache for the real BUNDLE_DIR.
    bundle = load_bundle()
    assert bundle.metadata["n_features"] == 131
