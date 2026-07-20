import json

from osu_ai_detector.cli import main


def _chart() -> str:
    objects = [f"{100 + i % 19},{90 + i % 17},{1000 + i * 250},1,0,0:0:0:0:" for i in range(45)]
    return "\n".join([
        "osu file format v14",
        "[General]",
        "Mode:0",
        "[Metadata]",
        "Title:test",
        "Artist:test",
        "Creator:human",
        "Version:test",
        "[Difficulty]",
        "SliderMultiplier:1.4",
        "[TimingPoints]",
        "0,500,4,2,1,70,1,0",
        "[HitObjects]",
        *objects,
        "",
    ])


def test_cli_reports_missing_content_model_without_legacy_promotion(tmp_path, capsys):
    path = tmp_path / "map.osu"
    path.write_text(_chart(), encoding="utf-8")
    code = main([str(path), "--json", "--content-model", str(tmp_path / "missing.json")])
    payload = json.loads(capsys.readouterr().out)
    report = payload["reports"][0]
    assert code == 0
    assert report["verdict"] == "inconclusive"
    assert report["analysis_channels"]["content_v2"]["available"] is False
    assert report["analysis_channels"]["public_revision_registry"]["known_exact_positive"] is False
    cheap = report["analysis_channels"]["cheap_fusion_v1"]
    assert cheap["status"] == "abstain"
    assert cheap["decision_usable"] is False
    assert report["analysis_channels"]["deprecated_exploratory_v0_2"]["used_for_final_verdict"] is False


def test_cli_can_explicitly_disable_public_revision_registry(tmp_path, capsys):
    path = tmp_path / "map.osu"
    path.write_text(_chart(), encoding="utf-8")
    main([str(path), "--json", "--no-revision-registry", "--content-model", str(tmp_path / "missing.json")])
    registry = json.loads(capsys.readouterr().out)["reports"][0]["analysis_channels"][
        "public_revision_registry"
    ]
    assert registry["status"] == "disabled"
    assert registry["available"] is False
    assert registry["known_exact_positive"] is False


def test_cli_can_explicitly_disable_cheap_fusion_without_disabling_fallback(tmp_path, capsys):
    path = tmp_path / "map.osu"
    path.write_text(_chart(), encoding="utf-8")
    main([str(path), "--json", "--no-fusion-model", "--content-model", str(tmp_path / "missing.json")])
    payload = json.loads(capsys.readouterr().out)
    fusion = payload["reports"][0]["analysis_channels"]["cheap_fusion_v1"]
    assert fusion["status"] == "disabled"
    assert fusion["decision_usable"] is False
    assert fusion["used_for_final_verdict"] is False
    assert payload["reports"][0]["verdict"] == "inconclusive"


def test_cli_whitebox_without_audio_returns_channel_error_without_loading_checkpoint(tmp_path, capsys):
    path = tmp_path / "map.osu"
    path.write_text(_chart(), encoding="utf-8")
    main([
        str(path),
        "--json",
        "--content-model",
        str(tmp_path / "missing.json"),
        "--whitebox",
        "--whitebox-checkpoint",
        "v31",
    ])
    payload = json.loads(capsys.readouterr().out)
    channel = payload["reports"][0]["analysis_channels"]["whitebox"]
    assert channel["status"] == "error"
    assert channel["error"]["type"] == "AudioRequired"


def test_cli_missing_whitebox_discriminator_is_explicit_abstention_not_zero(tmp_path, capsys):
    path = tmp_path / "map.osu"
    path.write_text(_chart(), encoding="utf-8")
    missing = tmp_path / "missing-whitebox-model.json"
    main([
        str(path),
        "--json",
        "--content-model",
        str(tmp_path / "missing-content.json"),
        "--whitebox",
        "--whitebox-checkpoint",
        "v31",
        "--whitebox-discriminator-model",
        str(missing),
    ])
    payload = json.loads(capsys.readouterr().out)
    discriminator = payload["reports"][0]["analysis_channels"]["whitebox"]["discriminator"]
    assert discriminator["enabled"] is True
    assert discriminator["status"] == "unavailable"
    assert discriminator["available"] is False
    assert discriminator["aggregate"] is None
    assert discriminator["used_for_final_verdict"] is False
    assert str(missing) in discriminator["reason"]


def test_cli_deep_fusion_requires_whitebox_discriminator(tmp_path, capsys):
    path = tmp_path / "map.osu"
    path.write_text(_chart(), encoding="utf-8")
    code = main([str(path), "--deep-fusion"])
    assert code == 1
    assert "requires the white-box discriminator" in capsys.readouterr().err


def test_cli_missing_deep_fusion_is_explicit_post_audio_abstention(tmp_path, capsys):
    path = tmp_path / "map.osu"
    path.write_text(_chart(), encoding="utf-8")
    missing_whitebox = tmp_path / "missing-whitebox.json"
    missing_deep = tmp_path / "missing-deep.json"
    main(
        [
            str(path),
            "--json",
            "--content-model",
            str(tmp_path / "missing-content.json"),
            "--whitebox",
            "--whitebox-checkpoint",
            "v31",
            "--whitebox-discriminator-model",
            str(missing_whitebox),
            "--deep-fusion-model",
            str(missing_deep),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    deep = payload["reports"][0]["analysis_channels"]["deep_fusion_v1"]
    assert deep["enabled"] is True
    assert deep["status"] == "unavailable"
    assert deep["available"] is False
    assert deep["score"] is None
    assert deep["independent_deep_conclusion"] == "abstain"
    assert str(missing_deep) in deep["reason"]


def test_cli_whitebox_uses_bounded_candidate_selection_and_forward_batch(
    monkeypatch, tmp_path, capsys
):
    import osu_ai_detector.cli as cli

    observed = {}

    class FakeWhiteboxEngine:
        def __init__(self, options):
            observed["options"] = options

        def score(self, beatmap_path, audio_path, *, candidate_interval_selection=None):
            observed["candidate"] = candidate_interval_selection
            return {"status": "ok", "settings": {}, "checkpoints": []}

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(cli, "WhiteboxEngine", FakeWhiteboxEngine)
    path = tmp_path / "map.osu"
    path.write_text(_chart(), encoding="utf-8")
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fixture")

    code = cli.main(
        [
            str(path),
            "--json",
            "--content-model",
            str(tmp_path / "missing-content.json"),
            "--whitebox",
            "--whitebox-checkpoint",
            "v31",
            "--audio",
            str(audio),
            "--whitebox-start-ms",
            "1000",
            "--whitebox-end-ms",
            "8000",
            "--whitebox-max-windows",
            "2",
            "--whitebox-forward-batch-size",
            "4",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    selection = observed["candidate"]

    assert code == 0
    assert observed["options"].forward_batch_size == 4
    assert selection["search_mode"] == "bounded_custom_raw_exploration"
    assert selection["calibration_eligible"] is False
    assert selection["selection_policy"]["windows_per_interval"] == 1
    assert all(1000 <= start < end <= 8000 for start, end in selection["intervals_ms"])
    assert payload["reports"][0]["analysis_channels"]["whitebox"][
        "candidate_interval_selection"
    ] == selection
    assert observed["closed"] is True


def test_cli_rejects_out_of_range_whitebox_forward_batch(tmp_path, capsys):
    path = tmp_path / "map.osu"
    path.write_text(_chart(), encoding="utf-8")

    code = main([str(path), "--whitebox-forward-batch-size", "65"])

    assert code == 1
    assert "must be in [1, 64]" in capsys.readouterr().err
