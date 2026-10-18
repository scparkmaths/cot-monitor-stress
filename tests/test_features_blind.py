from cms.features.shape import ALL_FEATURE_NAMES, FEATURE_NAMES, ShapeExtractor


def test_feature_schema_is_fixed_numeric_and_content_blind():
    text = "Maybe inspect the config. 1. Check state. Actually reconsider this? ```sh\necho ok\n```"
    # No local models are loaded in this unit test; the model-derived slots are
    # still present as floats, so downstream schema is identical.
    ext = ShapeExtractor.__new__(ShapeExtractor)
    got = ShapeExtractor.extract(ext, text, model_features=False, compute_nll=False)
    assert tuple(got) == ALL_FEATURE_NAMES
    assert set(FEATURE_NAMES).issubset(got)
    assert all(type(v) is float for v in got.values())
    lowered = {k.lower() for k in got}
    assert not any(
        word in key
        for word in ("config", "echo", "reconsider", "maybe")
        for key in lowered
    )
