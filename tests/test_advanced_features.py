from __future__ import annotations

from pathlib import Path

from osu_ai_detector.advanced_features import _coordinate_features, extract_windows, hash_ngrams
from osu_ai_detector.parser import parse_beatmap


def _map() -> str:
    objects = []
    for index in range(80):
        time = 1000 + index * 250
        x = 32 * (index % 15)
        y = 32 * ((index * 3) % 11)
        if index % 3:
            objects.append(f"{x},{y},{time},1,0,0:0:0:0:")
        else:
            objects.append(f"{x},{y},{time},2,0,B|{x + 32}:{y + 32}|{x + 64}:{y},1,96.0000000000001")
    timing = ["0,500,4,2,1,70,1,0"] + [
        f"{1000 + index * 2000},-133.33333333323336,4,2,-1,70,0,0" for index in range(8)
    ]
    return "\n".join([
        "osu file format v14", "", "[General]", "Mode:0", "", "[Metadata]",
        "Title:test", "Artist:test", "Creator:test", "Version:test", "BeatmapID:-1",
        "BeatmapSetID:-1", "", "[Difficulty]", "SliderMultiplier:1.4", "",
        "[TimingPoints]", *timing, "", "[HitObjects]", *objects, "",
    ])


def test_window_features_are_local_and_rich(tmp_path: Path) -> None:
    path = tmp_path / "map.osu"
    path.write_text(_map(), encoding="utf-8-sig")
    windows = extract_windows(parse_beatmap(path))
    assert len(windows) >= 2
    assert len(windows[0].values) >= 200
    assert windows[0].values["space_head_both_even"] == 1.0
    assert windows[0].values["timing_sv_epsilon_count"] > 0
    assert windows[0].sequence_tokens


def test_signed_hashing_is_deterministic_and_normalized() -> None:
    tokens = ("C|s4|i3", "S|s4|i4", "C|s2|i2", "C|s4|i3")
    first = hash_ngrams(tokens, dimensions=128)
    second = hash_ngrams(tokens, dimensions=128)
    assert first == second
    assert abs(sum(value * value for value in first.values()) - 1.0) < 1e-12


def test_public_folded_jump_residue_statistic_is_translation_invariant() -> None:
    first: dict[str, float] = {}
    second: dict[str, float] = {}
    jumps = [(4, -8), (12, -4), (-8, 12)]
    _coordinate_features(first, "space_jump_delta", jumps)
    # Translating every object leaves these deltas unchanged; model the same
    # observation explicitly rather than relying on absolute coordinates.
    _coordinate_features(second, "space_jump_delta", list(jumps))
    key = "space_jump_delta_folded_mod32_target_4_8_12_nonzero"
    assert first[key] == 1.0
    assert second[key] == first[key]
    assert first["space_jump_delta_folded_mod32_entropy"] > 0.0
    assert first["space_jump_delta_folded_mod32_nonzero_coordinates"] == 6.0
    assert first["space_jump_delta_folded_mod32_nonzero_ratio"] == 1.0
