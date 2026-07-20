from __future__ import annotations

import time

from fastapi.testclient import TestClient

import osu_ai_detector
import osu_ai_detector.web as web
from osu_ai_detector.web import create_app


def _chart() -> str:
    objects = "\n".join(f"{64 + index % 8 * 32},{96 + index % 4 * 32},{1000 + index * 250},1,0,0:0:0:0:" for index in range(80))
    return f"""osu file format v14

[General]
Mode:0

[Metadata]
Title:Release smoke test
Artist:NettoAndTetto
Creator:NettoAndTetto
Version:Normal

[Difficulty]
SliderMultiplier:1.4

[TimingPoints]
0,500,4,2,1,70,1,0

[HitObjects]
{objects}
"""


def test_release_version_and_english_local_ui() -> None:
    assert osu_ai_detector.__version__ == "1.0.0"
    with TestClient(create_app()) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert 'lang="en"' in response.text
    for label in (
        "Source Provenance Check",
        "Mapperatorinator Model Agreement",
        "Generator Trace Check",
        "Mapping Structure Check",
    ):
        assert label in response.text
    assert "no overall verdict" in response.text
    assert "Download technical JSON" in response.text
    assert "Download HTML report" in response.text


def test_release_source_only_job_is_streamed_without_overall_verdict() -> None:
    with TestClient(create_app()) as client:
        created = client.post(
            "/api/jobs",
            files=[("files", ("smoke.osu", _chart().encode(), "text/plain"))],
            data={"selected_methods": '["source"]', "whitebox_config": '{"enabled":false}'},
        )
        assert created.status_code == 202
        job = created.json()
        for _ in range(200):
            status = client.get(job["status_url"]).json()
            if status["progress"]["status"] in {"complete", "failed", "cancelled"}:
                break
            time.sleep(0.01)
        assert status["progress"]["status"] == "complete"
        result = client.get(job["results_url"]).json()["results"][0]
    assert set(result["methods"]) == {"source"}
    assert "verdict" not in result


def test_release_percentile_and_reason_contracts_are_human_readable() -> None:
    report = {
        "path": "map.osu",
        "analysis_channels": {
            "content_v2": {
                "available": True,
                "decision_usable": True,
                "score": 0.4,
                "human_null_p_value": 0.01,
                "thresholds": {"elevated": 0.5, "high": 0.8},
                "threshold_flags": {"elevated": False, "high": False},
                "top_evidence": [],
            }
        },
    }
    card = web._compact_method(report, 0, "content")
    assert card["percentile"] == 99.0
    assert card["state"] == "near"
    assert card["tone"] == "yellow"
    assert card["label"] == "Near threshold"

    reasons = web._positive_reasons(
        "forensic",
        {
            "source_anchored_positive_evidence": [
                {
                    "feature": "space_anchor_entropy_mod4",
                    "family": "decoder_coordinate_grid",
                    "value": 0.0,
                    "contribution": 0.0959,
                }
            ]
        },
    )
    assert reasons[0]["label"] == "Coordinate-grid trace"
    assert reasons[0]["observation"] == "Grid-residue entropy 0.0000"
    assert reasons[0]["contribution"] == 0.0959
    assert "not raw value" in reasons[0]["detail"]
