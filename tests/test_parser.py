from pathlib import Path

from osu_ai_detector.parser import parse_beatmap


def test_parser_accepts_utf8_bom_and_blank_lines_before_format_header(tmp_path: Path):
    chart = tmp_path / "legacy-preamble.osu"
    chart.write_bytes(
        b"\xef\xbb\xbf\r\n\r\nosu file format v14\r\n\r\n"
        b"[General]\r\nMode: 0\r\n\r\n"
        b"[Metadata]\r\nBeatmapID:797130\r\nBeatmapSetID:360118\r\n"
    )

    parsed = parse_beatmap(chart)

    assert parsed.format_version == 14
    assert parsed.had_utf8_bom is True
    assert parsed.metadata["BeatmapID"] == "797130"
