from __future__ import annotations

import copy
from pathlib import Path

import pytest

from osu_ai_detector.interval_selection import build_bound_candidate_selection
from osu_ai_detector.parser import Beatmap, HitObject
from osu_ai_detector.whitebox import (
    SCHEMA_VERSION as WHITEBOX_SCHEMA_VERSION,
    WhiteboxEngine,
    WhiteboxOptions,
)
from osu_ai_detector.whitebox_protocol import (
    SEARCH_PROTOCOL_SCHEMA_VERSION,
    canonical_search_protocol,
    derive_search_protocol,
    search_protocol_sha256,
)


CONTENT_SHA256 = "a" * 64


def _beatmap() -> Beatmap:
    return Beatmap(
        path=Path("memory.osu"),
        format_version=14,
        sections={},
        properties={"General": {"Mode": "0"}},
        timing_points=(),
        hit_objects=tuple(
            HitObject(100, 100, time, 1, 0, "circle") for time in range(0, 180_001, 5_000)
        ),
        raw_text="",
        had_utf8_bom=False,
    )


def _selection(*, windows_per_interval: int = 1) -> dict:
    return build_bound_candidate_selection(
        _beatmap(),
        {
            "model_id": "content-fixture-v1",
            "calibrated": True,
            "windows": [
                {"start_ms": 80_000, "end_ms": 104_000, "score": 0.9},
                {"start_ms": 120_000, "end_ms": 144_000, "score": 0.8},
            ],
        },
        content_model_sha256=CONTENT_SHA256,
        windows_per_interval=windows_per_interval,
    )


def _result(*, windows_per_interval: int = 1) -> dict:
    selection = _selection(windows_per_interval=windows_per_interval)
    return {
        "schema_version": WHITEBOX_SCHEMA_VERSION,
        "status": "ok",
        "settings": {
            "temperature": 0.9,
            "top_p": 0.9,
            "precision": "bf16",
            "attention_implementation": "sdpa",
            "start_ms": None,
            "end_ms": None,
            "intervals_ms": copy.deepcopy(selection["intervals_ms"]),
            "windows_per_interval": windows_per_interval,
            "forward_batch_size": 1,
            "max_windows": None,
        },
        "candidate_interval_selection": selection,
        "checkpoints": [],
    }


def test_bound_selection_and_derived_protocol_are_canonical_and_stable() -> None:
    result = _result()
    protocol = derive_search_protocol(result)
    assert protocol["schema_version"] == SEARCH_PROTOCOL_SCHEMA_VERSION
    assert protocol["candidate_interval_selection"]["content_model_id"] == "content-fixture-v1"
    assert protocol["candidate_interval_selection"]["content_model_sha256"] == CONTENT_SHA256
    assert protocol["execution"]["precision"] == "bf16"
    assert protocol["execution"]["attention_implementation"] == "sdpa"
    assert protocol["execution"]["forward_batch_size"] == 1
    assert canonical_search_protocol(protocol) == protocol
    assert search_protocol_sha256(copy.deepcopy(protocol)) == search_protocol_sha256(protocol)


@pytest.mark.parametrize("field,value", [("start_ms", 1), ("end_ms", 2), ("max_windows", 1)])
def test_custom_range_or_global_search_cap_fails_closed(field: str, value: int) -> None:
    result = _result()
    result["settings"][field] = value
    with pytest.raises(ValueError, match=field):
        derive_search_protocol(result)


def test_candidate_and_actual_intervals_must_match_exactly() -> None:
    result = _result()
    result["settings"]["intervals_ms"][0][1] += 1
    with pytest.raises(ValueError, match="does not exactly match"):
        derive_search_protocol(result)


def test_selection_policy_and_actual_window_multiplicity_must_match() -> None:
    result = _result(windows_per_interval=1)
    result["settings"]["windows_per_interval"] = 2
    with pytest.raises(ValueError, match="windows_per_interval"):
        derive_search_protocol(result)


@pytest.mark.parametrize("missing", ["precision", "attention_implementation"])
def test_runtime_math_fields_are_required(missing: str) -> None:
    result = _result()
    result["settings"].pop(missing)
    with pytest.raises(ValueError, match=missing.replace("_", " ")):
        derive_search_protocol(result)


def test_bound_selection_rejects_unbound_content_model() -> None:
    with pytest.raises(ValueError, match="content_model_sha256"):
        build_bound_candidate_selection(
            _beatmap(),
            {"model_id": "content-fixture-v1", "windows": []},
            content_model_sha256="not-a-digest",
        )


def test_engine_per_score_selection_is_attached_and_base_options_are_restored(tmp_path: Path) -> None:
    options = WhiteboxOptions(vendor_root=tmp_path / "missing-vendor")
    engine = WhiteboxEngine(options)
    selection = _selection(windows_per_interval=2)
    result = engine.score(
        tmp_path / "missing.osu",
        tmp_path / "missing.mp3",
        candidate_interval_selection=selection,
    )
    assert result["candidate_interval_selection"] == selection
    assert result["settings"]["intervals_ms"] == selection["intervals_ms"]
    assert result["settings"]["windows_per_interval"] == 2
    assert result["settings"]["attention_implementation"] == "sdpa"
    assert engine.options == options
