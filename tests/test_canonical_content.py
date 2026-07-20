from pathlib import Path

import pytest

from osu_ai_detector.canonical import (
    ContentWindow,
    canonical_sequence_features,
    canonical_sequence_tokens,
    feature_channel,
    hash_canonical_ngrams,
    representation_audit,
)
from osu_ai_detector.content_model import ContentEnsemble
from osu_ai_detector.interval_selection import merge_intervals, select_whitebox_intervals
from osu_ai_detector.parser import Beatmap, HitObject, TimingPoint


def _beatmap(objects):
    return Beatmap(
        path=Path("memory.osu"),
        format_version=14,
        sections={},
        properties={"General": {"Mode": "0"}},
        timing_points=(TimingPoint(0, 500, 4, 1, 0, 100, True, 0, "500"),),
        hit_objects=tuple(objects),
        raw_text="",
        had_utf8_bom=False,
    )


def _objects(transform=lambda x, y: (x, y)):
    rows = [
        (100, 100, 0, "circle", ()),
        (180, 100, 250, "slider", ((220, 140), (260, 100))),
        (260, 200, 500, "circle", ()),
    ]
    result = []
    for x, y, time, kind, anchors in rows:
        tx, ty = transform(x, y)
        transformed_anchors = tuple(transform(ax, ay) for ax, ay in anchors)
        result.append(
            HitObject(
                x=tx,
                y=ty,
                time=time,
                type_flags=2 if kind == "slider" else 1,
                hit_sound=0,
                kind=kind,
                anchors=transformed_anchors,
                curve_type="B" if kind == "slider" else None,
                repeats=1,
                pixel_length=140 if kind == "slider" else None,
            )
        )
    return result


def test_canonical_tokens_ignore_translation_rotation_and_reflection():
    original = _beatmap(_objects())
    translated = _beatmap(_objects(lambda x, y: (x + 40, y + 20)))
    rotated = _beatmap(_objects(lambda x, y: (512 - x, 384 - y)))
    reflected = _beatmap(_objects(lambda x, y: (512 - x, y)))
    expected = canonical_sequence_tokens(original, original.hit_objects)
    assert canonical_sequence_tokens(translated, translated.hit_objects) == expected
    assert canonical_sequence_tokens(rotated, rotated.hit_objects) == expected
    assert canonical_sequence_tokens(reflected, reflected.hit_objects) == expected


def test_allow_list_fails_closed_on_serialization_fields():
    assert feature_channel("file_utf8_bom") == "excluded"
    assert feature_channel("timing_sample_index_minus1") == "excluded"
    assert feature_channel("slider_decimal_mean") == "excluded"
    assert feature_channel("new_future_feature") == "excluded"
    assert feature_channel("rhythm_interval_beats_mean") == "semantic"
    assert feature_channel("space_head_mod32_16") == "mechanical"
    assert feature_channel("space_jump_delta_folded_mod32_target_4_8_12_nonzero") == "mechanical"
    assert feature_channel("timing_sv_quantized_001") == "mechanical"
    assert feature_channel("timing_red_green_same_offset") == "mechanical"
    audit = representation_audit()
    assert audit["raw_text_used"] is False
    assert audit["metadata_used"] is False


def test_canonical_hash_is_stable_and_normalized():
    values = hash_canonical_ngrams(("C|a", "S|b", "C|a"), dimensions=32)
    assert values == hash_canonical_ngrams(("C|a", "S|b", "C|a"), dimensions=32)
    assert sum(value * value for value in values.values()) == pytest.approx(1)


def test_sequence_complexity_exposes_repetition_without_coordinates():
    repeated = canonical_sequence_features(("C|a", "S|b", "C|a", "S|b", "C|a", "S|b"))
    varied = canonical_sequence_features(("C|a", "S|b", "P|c", "C|d", "S|e", "P|f"))
    assert repeated["sequence::bigram_unique_ratio"] < varied["sequence::bigram_unique_ratio"]
    assert repeated["sequence::lag_2_match_ratio"] > varied["sequence::lag_2_match_ratio"]
    assert repeated["sequence::lz78_phrase_ratio"] <= 1.0


def test_pure_python_tree_runtime_reports_path_evidence_and_ood():
    artifact = {
        "schema_version": 2,
        "model_id": "toy",
        "feature_names": ["object_count"],
        "ngram_dimensions": 8,
        "trees": [
            {
                "feature": [0, -2, -2],
                "threshold": [10.0, -2.0, -2.0],
                "left": [1, -1, -1],
                "right": [2, -1, -1],
                "probability": [0.5, 0.1, 0.9],
            }
        ],
        "ood_reference": {"median": [8.0], "robust_scale": [2.0], "abstain_distance": 5.0},
        "calibration": {"human_scores": [0.1, 0.2, 0.3], "thresholds": {}},
    }
    model = ContentEnsemble(artifact)
    window = ContentWindow(0, 1000, 20, {"object_count": 20.0}, {}, ())
    result = model.score_window(window)
    assert result.score == pytest.approx(0.9)
    assert result.ood_distance == pytest.approx(6.0)
    assert result.top_evidence[0]["feature"] == "object_count"
    assert result.top_evidence[0]["contribution"] == pytest.approx(0.4)


def test_runtime_reads_np_guarantee_objects_and_uses_strict_threshold():
    objects = [
        HitObject(100 + index % 10, 100, index * 100, 1, 0, "circle")
        for index in range(20)
    ]
    artifact = {
        "schema_version": 2,
        "model_id": "toy-calibrated",
        "feature_names": ["object_count"],
        "ngram_dimensions": 8,
        "window_ms": 24_000,
        "stride_ms": 8_000,
        "min_objects": 12,
        "aggregation_top_windows": 1,
        "trees": [
            {
                "feature": [-2],
                "threshold": [-2.0],
                "left": [-1],
                "right": [-1],
                "probability": [0.9],
            }
        ],
        "ood_reference": {"median": [20.0], "robust_scale": [1.0], "abstain_distance": 5.0},
        "calibration": {
            "human_scores": [0.1, 0.2, 0.3],
            "minimum_conformal_p": 0.25,
            "thresholds": {
                "elevated_np_fpr_1pct_delta_5pct": {
                    "supported": True, "threshold": 0.8, "operator": ">"
                },
                "high_np_fpr_0_1pct_delta_5pct": {
                    "supported": True, "threshold": 0.9, "operator": ">"
                },
            },
        },
    }
    result = ContentEnsemble(artifact).analyze(_beatmap(objects))
    assert result["calibrated"] is True
    assert result["threshold_flags"]["elevated"] is True
    assert result["threshold_flags"]["high"] is False  # equality is not a detection
    assert result["threshold_guarantees"]["high_np_fpr_0_1pct_delta_5pct"]["operator"] == ">"


def test_whitebox_interval_selection_is_label_free_and_merges_overlap():
    objects = [HitObject(100, 100, time, 1, 0, "circle") for time in range(0, 180_001, 5_000)]
    content = {
        "model_id": "toy",
        "calibrated": True,
        "windows": [
            {"start_ms": 80_000, "end_ms": 104_000, "score": 0.9},
            {"start_ms": 88_000, "end_ms": 112_000, "score": 0.89},  # redundant
            {"start_ms": 120_000, "end_ms": 144_000, "score": 0.8},
        ],
    }
    result = select_whitebox_intervals(_beatmap(objects), content, content_windows=2)
    assert result["label_free"] is True
    assert result["content_model_id"] == "toy"
    assert result["total_selected_ms"] > 0
    reasons = {reason for item in result["interval_details"] for reason in item["reasons"]}
    assert {"first_objects", "last_objects", "densest_objects", "content_top_1"} <= reasons
    assert "content_top_3" not in reasons


def test_merge_intervals_preserves_all_reasons():
    merged = merge_intervals([(0, 10, "a"), (12, 20, "b"), (100, 110, "c")], join_gap_ms=2)
    assert [(item.start_ms, item.end_ms) for item in merged] == [(0, 20), (100, 110)]
    assert merged[0].reasons == ("a", "b")
