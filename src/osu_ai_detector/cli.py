from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .detector import (
    DEFAULT_CONTENT_MODEL,
    DEFAULT_DEEP_FUSION_MODEL,
    DEFAULT_FORENSIC_MODEL,
    DEFAULT_FUSION_MODEL,
    DEFAULT_REVISION_REGISTRY,
    Detector,
    Verdict,
)
from .interval_selection import (
    build_bound_candidate_selection,
    build_candidate_selection_policy,
    select_whitebox_intervals,
)
from .parser import BeatmapParseError, parse_beatmap
from .whitebox import DEFAULT_VENDOR_ROOT, WhiteboxCheckpoint, WhiteboxEngine, WhiteboxOptions
from .whitebox_model import (
    DEFAULT_MODEL as DEFAULT_WHITEBOX_DISCRIMINATOR_MODEL,
    WhiteboxDiscriminator,
    unavailable as whitebox_discriminator_unavailable,
)


MAX_FORWARD_BATCH_SIZE = 64


def _paths(inputs: list[str], recursive: bool) -> list[Path]:
    result: list[Path] = []
    for token in inputs:
        path = Path(token)
        if path.is_file() and path.suffix.lower() == ".osu":
            result.append(path)
        elif path.is_dir():
            iterator = path.rglob("*.osu") if recursive else path.glob("*.osu")
            result.extend(iterator)
    return sorted(set(path.resolve() for path in result))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osu-ai-detect",
        description="Content-v2 detector with optional Mapperatorinator white-box analysis.",
    )
    parser.add_argument("inputs", nargs="+", help=".osu files or directories")
    parser.add_argument("--recursive", "-r", action="store_true", help="scan directories recursively")
    parser.add_argument("--json", action="store_true", help="emit one JSON document")
    parser.add_argument("--jsonl", action="store_true", help="emit one JSON object per map")
    parser.add_argument("--no-features", action="store_true", help="omit the full feature vector from JSON")
    parser.add_argument("--compact", action="store_true", help="show only verdict, score and path in text mode")
    parser.add_argument("--pretty", action="store_true", help="explicitly request the default detailed text report")
    parser.add_argument(
        "--content-model",
        type=Path,
        default=DEFAULT_CONTENT_MODEL,
        help="content_v2 JSON artifact (missing is reported as unavailable; no silent fallback)",
    )
    parser.add_argument("--no-content-model", action="store_true", help="disable content_v2 and explicitly abstain")
    parser.add_argument(
        "--forensic-model",
        type=Path,
        default=DEFAULT_FORENSIC_MODEL,
        help="revision_forensics_v3 JSON artifact (missing/uncalibrated explicitly abstains)",
    )
    parser.add_argument("--no-forensic-model", action="store_true", help="disable revision_forensics_v3")
    parser.add_argument(
        "--revision-registry",
        type=Path,
        default=DEFAULT_REVISION_REGISTRY,
        help="exact-checksum public historical revision registry",
    )
    parser.add_argument("--no-revision-registry", action="store_true", help="disable public revision lookup")
    parser.add_argument(
        "--fusion-model",
        type=Path,
        default=DEFAULT_FUSION_MODEL,
        help="independently calibrated content_v2 + revision_forensics_v3 cheap-fusion JSON artifact",
    )
    parser.add_argument("--no-fusion-model", action="store_true", help="disable cheap fusion and use base fallback")
    parser.add_argument("--whitebox", action="store_true", help="enable optional Mapperatorinator checkpoint scoring")
    parser.add_argument(
        "--whitebox-checkpoint",
        action="append",
        choices=("v29", "v30", "v31", "v32", "v32-mini"),
        help="checkpoint config to score; repeatable (default: all five)",
    )
    parser.add_argument("--audio", type=Path, help="audio shared by all input maps; otherwise use each AudioFilename")
    parser.add_argument("--whitebox-vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT)
    parser.add_argument("--whitebox-device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--whitebox-precision", choices=("fp32", "bf16", "amp"), default="bf16")
    parser.add_argument("--whitebox-start-ms", type=int)
    parser.add_argument("--whitebox-end-ms", type=int)
    parser.add_argument("--whitebox-max-windows", type=int)
    parser.add_argument("--whitebox-token-details", action="store_true")
    parser.add_argument("--whitebox-token-limit", type=int, default=256)
    parser.add_argument("--whitebox-temperature", type=float, default=0.9)
    parser.add_argument("--whitebox-top-p", type=float, default=0.9)
    parser.add_argument(
        "--whitebox-forward-batch-size",
        type=int,
        default=1,
        metavar="N",
        help=(
            "teacher-forcing batch size in [1, 64] (default: 1); changing it keeps raw diagnostics "
            "but normally invalidates a calibrated search-protocol match"
        ),
    )
    parser.add_argument(
        "--whitebox-discriminator",
        action="store_true",
        help="apply the optional CPU JSON discriminator after raw white-box scoring",
    )
    parser.add_argument(
        "--whitebox-discriminator-model",
        type=Path,
        help=(
            "white-box discriminator JSON artifact; specifying a path enables it. The absent production default "
            f"is {DEFAULT_WHITEBOX_DISCRIMINATOR_MODEL}"
        ),
    )
    parser.add_argument(
        "--deep-fusion",
        action="store_true",
        help=(
            "after white-box discriminator scoring, apply the independently calibrated "
            "cheap+white-box whole-map statistic"
        ),
    )
    parser.add_argument(
        "--deep-fusion-model",
        type=Path,
        help=(
            "deep-fusion JSON artifact; specifying a path enables it. The absent production default "
            f"is {DEFAULT_DEEP_FUSION_MODEL}"
        ),
    )
    return parser


def _print_content(content: dict) -> None:
    if not content.get("available"):
        print(f"  content_v2: unavailable/abstain ({content.get('reason', 'unknown reason')})")
        print(f"    model_path={content.get('model_path')}")
        return
    ood = content.get("ood", {})
    print(
        "  content_v2: "
        f"model={content.get('model_id')} score={content.get('score', 0):.6f} "
        f"human_null_p={content.get('human_null_p_value')} calibrated={content.get('calibrated')} "
        f"decision_usable={content.get('decision_usable')} ood_abstain={ood.get('abstain')}"
    )
    print(f"    thresholds={json.dumps(content.get('thresholds', {}), ensure_ascii=False, sort_keys=True)}")
    print(f"    ood={json.dumps(ood, ensure_ascii=False, sort_keys=True)}")
    for item in content.get("top_evidence", [])[:12]:
        print(
            f"    evidence {item.get('window_start_ms')}-{item.get('window_end_ms')}ms "
            f"{item.get('feature')} value={item.get('value')} contribution={item.get('contribution'):+.6f} "
            f"robust_z={item.get('robust_z')}"
        )
    for window in content.get("windows", []):
        print(
            f"    window {window.get('start_ms')}-{window.get('end_ms')}ms "
            f"objects={window.get('object_count')} score={window.get('score', 0):.6f} "
            f"ood={window.get('ood_distance', 0):.4f}"
        )


def _print_forensic(forensic: dict) -> None:
    if not forensic.get("available"):
        print(
            "  revision_forensics_v3: unavailable/abstain "
            f"({forensic.get('reason') or forensic.get('abstention_reason') or 'unknown reason'})"
        )
        print(f"    model_path={forensic.get('model_path')}")
        return
    print(
        "  revision_forensics_v3: "
        f"status={forensic.get('status')} model={forensic.get('model_id')} score={forensic.get('score', 0):.6f} "
        f"human_null_p={forensic.get('human_null_p_value')} calibrated={forensic.get('calibrated')} "
        f"decision_usable={forensic.get('decision_usable')} role={forensic.get('channel_role')}"
    )
    print(
        f"    thresholds={json.dumps(forensic.get('thresholds', {}), ensure_ascii=False, sort_keys=True)} "
        f"flags={json.dumps(forensic.get('threshold_flags', {}), ensure_ascii=False, sort_keys=True)}"
    )
    if forensic.get("abstention_reason"):
        print(f"    abstention={forensic.get('abstention_reason')}")
    for item in forensic.get("top_evidence", [])[:12]:
        print(
            f"    evidence {item.get('window_start_ms')}-{item.get('window_end_ms')}ms "
            f"{item.get('family')}/{item.get('feature')} value={item.get('value')} "
            f"contribution={item.get('contribution', 0):+.6f} source_anchored={item.get('source_anchored')}"
        )


def _print_revision_registry(registry: dict) -> None:
    if not registry.get("available"):
        print(f"  public_revision_registry: unavailable ({registry.get('reason', 'unknown reason')})")
        print(f"    registry_path={registry.get('registry_path')}")
        return
    observed = registry.get("observed") or {}
    print(
        "  public_revision_registry: "
        f"status={registry.get('status')} known_exact_positive={registry.get('known_exact_positive')} "
        f"used_for_final_verdict={registry.get('used_for_final_verdict')}"
    )
    print(
        f"    registry={registry.get('registry_id')} path={registry.get('registry_path')} "
        f"observed_sha256={observed.get('sha256')} beatmap_id={observed.get('beatmap_id')} "
        f"beatmapset_id={observed.get('beatmapset_id')}"
    )
    print(f"    reason={registry.get('reason')}")
    if registry.get("exact_match"):
        print(f"    exact_match={json.dumps(registry.get('exact_match'), ensure_ascii=False, sort_keys=True)}")
    if registry.get("identity_history") or registry.get("missing_revision_history"):
        print(
            "    non-inheriting history="
            + json.dumps(
                {
                    "identity_history": registry.get("identity_history"),
                    "missing_revision_history": registry.get("missing_revision_history"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def _print_fusion(fusion: dict) -> None:
    if not fusion.get("available"):
        print(f"  cheap_fusion_v1: unavailable/abstain ({fusion.get('reason', 'unknown reason')})")
        print(f"    model_path={fusion.get('model_path')} observed_hashes={fusion.get('observed_base_model_sha256')}")
        return
    print(
        "  cheap_fusion_v1: "
        f"status={fusion.get('status')} model={fusion.get('model_id')} score={fusion.get('score')} "
        f"human_null_p={fusion.get('human_null_p_value')} calibrated={fusion.get('calibrated')} "
        f"decision_usable={fusion.get('decision_usable')} used={fusion.get('used_for_final_verdict')}"
    )
    print(
        f"    thresholds={json.dumps(fusion.get('thresholds', {}), ensure_ascii=False, sort_keys=True)} "
        f"flags={json.dumps(fusion.get('threshold_flags', {}), ensure_ascii=False, sort_keys=True)}"
    )
    print(
        "    binding="
        + json.dumps(
            {
                "expected": fusion.get("base_model_binding"),
                "observed_sha256": fusion.get("observed_base_model_sha256"),
                "whitebox_included": fusion.get("whitebox_included"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if fusion.get("abstention_reasons"):
        print(f"    abstention_reasons={json.dumps(fusion.get('abstention_reasons'), ensure_ascii=False)}")
    for item in fusion.get("channels", []):
        print(
            f"    {item.get('channel')}: raw={item.get('raw_score')} "
            f"dev_tail_p={item.get('development_upper_tail_p')} "
            f"surprisal={item.get('development_tail_surprisal')} weight={item.get('weight')} "
            f"contribution={item.get('weighted_contribution')}"
        )


def _print_whitebox_discriminator(discriminator: dict) -> None:
    if not discriminator.get("enabled"):
        print("    discriminator: not requested")
        return
    if not discriminator.get("available"):
        print(
            "    discriminator: unavailable/abstain "
            f"({discriminator.get('reason', 'unknown reason')}) model_path={discriminator.get('model_path')}"
        )
        return
    aggregate = discriminator.get("aggregate") or {}
    calibration = discriminator.get("calibration") or {}
    protocol = discriminator.get("calibration_protocol_audit") or {}
    print(
        "    discriminator: "
        f"model={discriminator.get('model_id')} ranking_score={aggregate.get('ranking_score')} "
        f"status={discriminator.get('status')} decision_usable={discriminator.get('decision_usable')} "
        f"uncalibrated={discriminator.get('uncalibrated')} human_null_p={calibration.get('human_null_p_value')} "
        f"used_for_final_verdict={discriminator.get('used_for_final_verdict')}"
    )
    print(
        "      calibrated_protocol: "
        f"exact_match={protocol.get('exact_match')} "
        f"required={protocol.get('required_checkpoints')} "
        f"observed={protocol.get('observed_checkpoint_entries')} "
        f"missing={protocol.get('missing_checkpoints')} extra={protocol.get('extra_checkpoints')} "
        f"duplicates={protocol.get('duplicate_checkpoints')} identities_exact={protocol.get('checkpoint_identities_exact')} "
        f"search_exact={protocol.get('search_protocol_exact')}"
    )
    print(
        "      search_protocol_hashes: "
        f"required={protocol.get('required_search_protocol_sha256')} "
        f"observed={protocol.get('observed_search_protocol_sha256')}"
    )
    if protocol.get("search_protocol_errors"):
        print(
            "      search_protocol_errors="
            + json.dumps(protocol.get("search_protocol_errors"), ensure_ascii=False, sort_keys=True)
        )
    if protocol.get("search_protocol_mismatches"):
        print(
            "      search_protocol_mismatches="
            + json.dumps(protocol.get("search_protocol_mismatches"), ensure_ascii=False, sort_keys=True)
        )
    if protocol.get("reasons"):
        print(f"      protocol_abstention_reasons={json.dumps(protocol.get('reasons'), ensure_ascii=False)}")
    print(f"      selected_windows={json.dumps(aggregate.get('selected_windows', []), ensure_ascii=False)}")
    for window in discriminator.get("windows", []):
        print(
            f"      window {window.get('checkpoint')}#{window.get('window_index')} "
            f"{window.get('start_ms')}-{window.get('end_ms')}ms tokens={window.get('token_count')} "
            f"score={window.get('ranking_score'):.6f} logit={window.get('logit'):.6f}"
        )
        for item in window.get("top_contributions", [])[:5]:
            print(
                f"        {item.get('feature')} raw={item.get('raw_value')} missing={item.get('was_missing')} "
                f"scaled={item.get('scaled_value'):.6g} weight={item.get('weight'):.6g} "
                f"contribution={item.get('contribution'):+.6g}"
            )


def _print_deep_fusion(deep: dict) -> None:
    if not deep.get("enabled"):
        print("  deep_fusion_v1: not requested")
        return
    if not deep.get("available"):
        print(
            "  deep_fusion_v1: unavailable/abstain "
            f"({deep.get('reason', 'unknown reason')}) model_path={deep.get('model_path')}"
        )
        return
    print(
        "  deep_fusion_v1: "
        f"status={deep.get('status')} model={deep.get('model_id')} score={deep.get('score')} "
        f"human_null_p={deep.get('human_null_p_value')} calibrated={deep.get('calibrated')} "
        f"decision_usable={deep.get('decision_usable')} "
        f"conclusion={deep.get('independent_deep_conclusion')}"
    )
    print(
        f"    thresholds={json.dumps(deep.get('thresholds', {}), ensure_ascii=False, sort_keys=True)} "
        f"flags={json.dumps(deep.get('threshold_flags', {}), ensure_ascii=False, sort_keys=True)}"
    )
    print(
        "    binding="
        + json.dumps(
            {
                "expected": deep.get("base_model_binding"),
                "observed_sha256": deep.get("observed_base_model_sha256"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if deep.get("abstention_reasons"):
        print(f"    abstention_reasons={json.dumps(deep.get('abstention_reasons'), ensure_ascii=False)}")
    for item in deep.get("channels", []):
        print(
            f"    {item.get('channel')}: raw={item.get('raw_map_score')} "
            f"dev_tail_p={item.get('development_upper_tail_p')} "
            f"surprisal={item.get('development_tail_surprisal')} weight={item.get('fusion_weight')} "
            f"contribution={item.get('weighted_contribution')} "
            f"base_calibrated={item.get('base_calibrated')}"
        )


def _print_whitebox(whitebox: dict) -> None:
    if not whitebox.get("enabled"):
        print("  whitebox: not requested")
        return
    print(f"  whitebox: status={whitebox.get('status')} audio_source={whitebox.get('audio_source')}")
    if whitebox.get("status") not in {"ok", "partial"}:
        print(f"    detail={json.dumps(whitebox.get('error') or whitebox.get('availability'), ensure_ascii=False)}")
        _print_whitebox_discriminator(whitebox.get("discriminator", {}))
        return
    selection = whitebox.get("candidate_interval_selection") or {}
    if selection:
        print(
            "    candidate_search: "
            f"mode={selection.get('search_mode')} label_free={selection.get('label_free')} "
            f"calibration_eligible={selection.get('calibration_eligible')} "
            f"content_model_id={selection.get('content_model_id')} "
            f"content_model_sha256={selection.get('content_model_sha256')}"
        )
        print(
            f"      intervals_ms={json.dumps(selection.get('intervals_ms', []), ensure_ascii=False)} "
            f"policy={json.dumps(selection.get('selection_policy', {}), ensure_ascii=False, sort_keys=True)}"
        )
        if selection.get("protocol_notice"):
            print(f"      notice={selection.get('protocol_notice')}")
        if selection.get("protocol_incomplete_reason"):
            print(f"      incomplete_reason={selection.get('protocol_incomplete_reason')}")
        if selection.get("manual_search_overrides"):
            print(
                "      manual_overrides="
                + json.dumps(selection.get("manual_search_overrides"), ensure_ascii=False, sort_keys=True)
            )
        if selection.get("custom_range_filter"):
            print(
                "      custom_range_filter="
                + json.dumps(selection.get("custom_range_filter"), ensure_ascii=False, sort_keys=True)
            )
    for checkpoint in whitebox.get("checkpoints", []):
        print(
            f"    checkpoint={checkpoint.get('checkpoint')} status={checkpoint.get('status')} "
            f"windows={checkpoint.get('window_count', 0)} tokens={checkpoint.get('coverage', {}).get('token_count', 0)}"
        )
        if checkpoint.get("status") != "ok":
            print(f"      error={json.dumps(checkpoint.get('error'), ensure_ascii=False)}")
            continue
        for family, values in checkpoint.get("families", {}).items():
            coverage = values.get("coverage", {})
            nll = values.get("raw_full_vocabulary", {}).get("nll", {})
            rank = values.get("family_conditioned", {}).get("rank", {})
            curvature = values.get("family_conditioned", {}).get("curvature_z", {})
            violations = values.get("runs", {}).get("top_p_violations", {})
            policy = values.get("generation_policy", {})
            policy_support = policy.get("sampling_support", {})
            policy_runs = policy.get("runs", {}).get(
                "sampling_support_violations", {}
            )
            policy_lrr = values.get("detectllm_lrr", {}).get(
                "generation_policy_pre_truncation", {}
            )
            policy_fd = values.get(
                "fast_detectgpt_policy_aware_sequence_discrepancy", {}
            )
            print(
                f"      {family}: tokens={coverage.get('token_count')} windows={coverage.get('windows_with_tokens')} "
                f"raw_nll_mean/p50={nll.get('mean')}/{nll.get('p50')} "
                f"policy_coverage={policy.get('coverage_fraction')} "
                f"policy_support_fraction={policy_support.get('in_support_fraction')} "
                f"policy_violation_runs/longest={policy_runs.get('run_count')}/{policy_runs.get('longest_run')} "
                f"policy_lrr={policy_lrr.get('value')} policy_fast_detectgpt={policy_fd.get('value')} "
                f"target_family_rank_mean/p50={rank.get('mean')}/{rank.get('p50')} "
                f"target_family_nucleus_fraction={values.get('sampling', {}).get('in_nucleus_fraction')} "
                f"target_family_exclusion_runs/longest={violations.get('run_count')}/{violations.get('longest_run')} "
                f"target_family_token_z_mean/p50={curvature.get('mean')}/{curvature.get('p50')}"
            )
        for window in checkpoint.get("windows", []):
            print(
                f"      window {window.get('window_index')} "
                f"{window.get('scored_interval_start_ms')}-{window.get('scored_interval_end_ms')}ms "
                f"tokens={window.get('token_count')} families={','.join(window.get('families', {}))}"
            )
    _print_whitebox_discriminator(whitebox.get("discriminator", {}))


def _filter_cli_candidate_intervals(
    selection: dict,
    *,
    start_ms: int | None,
    end_ms: int | None,
) -> dict:
    if start_ms is None and end_ms is None:
        return selection
    filtered: list[list[int]] = []
    for raw in selection.get("intervals_ms", []):
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            continue
        start, end = int(raw[0]), int(raw[1])
        bounded_start = max(start, start_ms or 0)
        bounded_end = min(end, end_ms) if end_ms is not None else end
        if bounded_start < bounded_end:
            filtered.append([bounded_start, bounded_end])
    result = dict(selection)
    result["custom_range_filter"] = {
        "requested_start_ms": start_ms,
        "requested_end_ms": end_ms,
        "automatic_candidate_count": len(selection.get("intervals_ms", [])),
        "intersecting_candidate_count": len(filtered),
        "behavior_when_no_overlap": (
            "retain automatic proposals behind the engine range gate; score zero windows, never the whole song"
        ),
    }
    if filtered:
        result["intervals_ms"] = filtered
        result["interval_details"] = [
            {
                "start_ms": start,
                "end_ms": end,
                "reasons": ["automatic_candidate_intersected_with_custom_range"],
            }
            for start, end in filtered
        ]
        result["total_selected_ms"] = sum(end - start for start, end in filtered)
    return result


def _cli_candidate_selection(path: Path, report, args, *, content_model_sha256: str | None) -> dict:
    """Build a bounded, auditable candidate search for one CLI map."""

    beatmap = parse_beatmap(path)
    content = report.analysis_channels.get("content_v2", {})
    try:
        if not isinstance(content, dict) or content.get("available") is not True:
            raise ValueError("content analysis is unavailable")
        if not isinstance(content_model_sha256, str):
            raise ValueError("content artifact SHA-256 is unavailable")
        selection = build_bound_candidate_selection(
            beatmap,
            content,
            content_model_sha256=content_model_sha256,
            windows_per_interval=1,
        )
        selection["search_mode"] = "calibration_bound_label_free_auto_candidates"
        selection["calibration_eligible"] = True
        selection["protocol_notice"] = (
            "label-free automatic candidates are bound to the active content model ID and artifact SHA-256; "
            "the discriminator still requires an exact full-protocol match"
        )
    except (TypeError, ValueError) as exc:
        selection = select_whitebox_intervals(beatmap, None)
        if not selection.get("intervals_ms"):
            raise ValueError("cannot build a bounded label-free white-box search: beatmap has no scorable objects")
        selection.update(
            {
                "content_model_sha256": content_model_sha256,
                "selection_policy": build_candidate_selection_policy(1),
                "search_mode": "safe_label_free_fallback",
                "calibration_eligible": False,
                "protocol_notice": (
                    "content model identity is incomplete; raw scoring is limited to first/last/densest proposals, "
                    "never the whole song, and calibrated discrimination must abstain"
                ),
                "protocol_incomplete_reason": str(exc),
            }
        )
    overrides = {
        "start_ms": args.whitebox_start_ms,
        "end_ms": args.whitebox_end_ms,
        "max_windows": args.whitebox_max_windows,
    }
    if any(value is not None for value in overrides.values()):
        selection = _filter_cli_candidate_intervals(
            selection,
            start_ms=args.whitebox_start_ms,
            end_ms=args.whitebox_end_ms,
        )
        selection.update(
            {
                "search_mode": "bounded_custom_raw_exploration",
                "calibration_eligible": False,
                "manual_search_overrides": overrides,
                "protocol_notice": (
                    "manual range/window settings are preserved for bounded raw exploration; the non-null engine "
                    "settings intentionally force calibrated protocol abstention"
                ),
            }
        )
    return selection


def _print_detailed(report, whitebox: dict | None = None) -> None:
    identity = report.map_identity
    print(f"{report.verdict.value}  discriminator={report.evidence_score:.4f}")
    print(f"  file: {report.path}")
    print(
        "  identity: "
        f"b={identity.get('beatmap_id') or '-'} s={identity.get('beatmapset_id') or '-'} "
        f"{identity.get('artist') or '?'} - {identity.get('title') or '?'} "
        f"[{identity.get('version') or '?'}] by {identity.get('creator') or '?'}"
    )
    print(f"  md5: {identity.get('md5')}  sha256: {identity.get('sha256')}")
    _print_content(report.analysis_channels.get("content_v2", {}))
    _print_revision_registry(report.analysis_channels.get("public_revision_registry", {}))
    _print_fusion(report.analysis_channels.get("cheap_fusion_v1", {}))
    _print_forensic(report.analysis_channels.get("revision_forensics_v3", {}))
    _print_whitebox(whitebox or report.analysis_channels.get("whitebox", {}))
    if "deep_fusion_v1" in report.analysis_channels:
        _print_deep_fusion(report.analysis_channels.get("deep_fusion_v1", {}))
    print("  evidence:")
    if report.evidence:
        for item in report.evidence:
            print(
                f"    - {item.family}/{item.key}: contribution={item.contribution:.4f} "
                f"(strength={item.strength:.4f}, reliability={item.reliability:.4f})"
            )
            print(f"      {item.description}")
            print(f"      observed={json.dumps(item.observed, ensure_ascii=False, sort_keys=True)}")
    else:
        print("    - none")

    print("  deprecated exploratory v0.2 (not used for final verdict):")
    statistical = report.decision_trace.get("statistical_model", {})
    if statistical.get("available"):
        thresholds = statistical.get("thresholds", {})
        print(
            "  statistical: "
            f"model={statistical.get('model_id')} combined={statistical.get('combined_score', 0):.4f} "
            f"concordance={statistical.get('concordance_score', 0):.4f} "
            f"numeric={statistical.get('numeric_score', 0):.4f} "
            f"sequence={statistical.get('sequence_score', 0):.4f} "
            f"high={statistical.get('high')} suspicious={statistical.get('suspicious')}"
        )
        print(f"    settings={json.dumps(statistical.get('settings', {}), ensure_ascii=False, sort_keys=True)}")
        print(f"    thresholds={json.dumps(thresholds, ensure_ascii=False, sort_keys=True)}")
        for segment in statistical.get("segments", [])[:5]:
            print(
                f"    segment {segment['start_ms']/1000:.3f}-{segment['end_ms']/1000:.3f}s "
                f"objects={segment['object_count']} combined={segment['combined_score']:.4f} "
                f"concordance={segment['concordance_score']:.4f} "
                f"numeric={segment['numeric_score']:.4f} sequence={segment['sequence_score']:.4f}"
            )
            numeric = segment.get("top_numeric", [])[:3]
            if numeric:
                print("      numeric: " + "; ".join(
                    f"{item['feature']}={item['value']:.4g} ({item['contribution']:+.4f})" for item in numeric
                ))
            sequence = segment.get("top_sequence", [])[:3]
            if sequence:
                print("      sequence: " + "; ".join(
                    f"bin{item['bin']} ({item['contribution']:+.4f}) examples={item.get('examples', [])}"
                    for item in sequence
                ))
    else:
        print(f"  statistical: unavailable ({statistical.get('reason', 'unknown reason')})")
    print(f"  counts: {json.dumps(report.counts, ensure_ascii=False, sort_keys=True)}")
    for caveat in report.caveats:
        print(f"  caveat: {caveat}")
    print("  note: scores are not calibrated AI-authorship probabilities; v0.2 does not set the verdict.")


def _associated_audio(path: Path) -> Path | None:
    beatmap = parse_beatmap(path)
    filename = beatmap.properties.get("General", {}).get("AudioFilename", "").strip()
    if not filename:
        return None
    candidate = (path.parent / filename).resolve()
    try:
        candidate.relative_to(path.parent.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = _paths(args.inputs, args.recursive)
    if not paths:
        print("No .osu files found.", file=sys.stderr)
        return 1

    if not 1 <= args.whitebox_forward_batch_size <= MAX_FORWARD_BATCH_SIZE:
        print(
            f"--whitebox-forward-batch-size must be in [1, {MAX_FORWARD_BATCH_SIZE}]",
            file=sys.stderr,
        )
        return 1
    if args.whitebox_start_ms is not None and args.whitebox_start_ms < 0:
        print("--whitebox-start-ms cannot be negative", file=sys.stderr)
        return 1
    if args.whitebox_end_ms is not None and args.whitebox_end_ms <= 0:
        print("--whitebox-end-ms must be positive", file=sys.stderr)
        return 1
    if (
        args.whitebox_start_ms is not None
        and args.whitebox_end_ms is not None
        and args.whitebox_start_ms >= args.whitebox_end_ms
    ):
        print("--whitebox-start-ms must be less than --whitebox-end-ms", file=sys.stderr)
        return 1
    if args.whitebox_max_windows is not None and not 1 <= args.whitebox_max_windows <= 10000:
        print("--whitebox-max-windows must be in [1, 10000]", file=sys.stderr)
        return 1

    discriminator_enabled = bool(args.whitebox_discriminator or args.whitebox_discriminator_model is not None)
    deep_fusion_enabled = bool(args.deep_fusion or args.deep_fusion_model is not None)
    if discriminator_enabled and not args.whitebox:
        print("--whitebox-discriminator/--whitebox-discriminator-model requires --whitebox", file=sys.stderr)
        return 1
    if deep_fusion_enabled and not discriminator_enabled:
        print("--deep-fusion/--deep-fusion-model requires the white-box discriminator", file=sys.stderr)
        return 1

    detector = Detector(
        content_model_path=args.content_model,
        content_model_enabled=not args.no_content_model,
        forensic_model_path=args.forensic_model,
        forensic_model_enabled=not args.no_forensic_model,
        revision_registry_path=args.revision_registry,
        revision_registry_enabled=not args.no_revision_registry,
        fusion_model_path=args.fusion_model,
        fusion_model_enabled=not args.no_fusion_model,
        deep_fusion_model_path=args.deep_fusion_model or DEFAULT_DEEP_FUSION_MODEL,
        deep_fusion_model_enabled=deep_fusion_enabled,
    )
    discriminator_path = (
        args.whitebox_discriminator_model or DEFAULT_WHITEBOX_DISCRIMINATOR_MODEL
    ).expanduser().resolve()
    discriminator: WhiteboxDiscriminator | None = None
    discriminator_error: dict | None = None
    if discriminator_enabled:
        if not discriminator_path.is_file():
            discriminator_error = whitebox_discriminator_unavailable(
                f"white-box discriminator artifact is missing: {discriminator_path}; the channel abstains",
                model_path=discriminator_path,
            )
        else:
            try:
                discriminator = WhiteboxDiscriminator.from_path(discriminator_path)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                discriminator_error = whitebox_discriminator_unavailable(
                    f"cannot load white-box discriminator artifact: {type(exc).__name__}: {exc}",
                    model_path=discriminator_path,
                )

    def attach_discriminator(result: dict, report) -> None:
        if not discriminator_enabled:
            result["discriminator"] = {
                "enabled": False,
                "status": "not_requested",
                "available": False,
                "used_for_final_verdict": False,
            }
            return
        if discriminator is None:
            detail = dict(discriminator_error or whitebox_discriminator_unavailable("model unavailable"))
        else:
            try:
                detail = discriminator.score(result)
            except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as exc:
                detail = whitebox_discriminator_unavailable(
                    f"white-box discriminator scoring failed: {type(exc).__name__}: {exc}",
                    model_path=discriminator_path,
                )
        detail["enabled"] = True
        detail["model_path"] = str(discriminator_path)
        # This channel is diagnostic until a separate multi-channel fusion is calibrated.
        detail["used_for_final_verdict"] = False
        result["discriminator"] = detail
        if deep_fusion_enabled:
            deep_detail = detector.analyze_deep_fusion(
                report.analysis_channels.get("cheap_fusion_v1", {}),
                detail,
                whitebox_model_path=discriminator_path,
            )
            result["deep_fusion"] = deep_detail
            # DetectionReport is frozen, but its per-channel audit mapping is
            # deliberately mutable so post-audio channels can be attached.
            report.analysis_channels["deep_fusion_v1"] = deep_detail
    engine = None
    if args.whitebox:
        checkpoint_names = args.whitebox_checkpoint or ["v29", "v30", "v31", "v32", "v32-mini"]
        try:
            engine = WhiteboxEngine(WhiteboxOptions(
                checkpoints=tuple(WhiteboxCheckpoint(name) for name in checkpoint_names),
                vendor_root=args.whitebox_vendor_root,
                device=args.whitebox_device,
                precision=args.whitebox_precision,
                temperature=args.whitebox_temperature,
                top_p=args.whitebox_top_p,
                forward_batch_size=args.whitebox_forward_batch_size,
                start_ms=args.whitebox_start_ms,
                end_ms=args.whitebox_end_ms,
                max_windows=args.whitebox_max_windows,
                include_token_details=args.whitebox_token_details,
                max_token_details_per_window=args.whitebox_token_limit,
            ))
        except ValueError as exc:
            print(f"Invalid white-box configuration: {exc}", file=sys.stderr)
            return 1
    reports = []
    whitebox_results: dict[str, dict] = {}
    errors = []
    try:
        for path in paths:
            try:
                report = detector.analyze(path)
                reports.append(report)
                if engine is not None:
                    audio_path = args.audio.resolve() if args.audio else _associated_audio(path)
                    if audio_path is None or not audio_path.is_file():
                        whitebox_results[str(path)] = {
                            "enabled": True,
                            "status": "error",
                            "used_for_final_verdict": False,
                            "error": {"type": "AudioRequired", "message": "no usable audio; pass --audio or set AudioFilename"},
                        }
                        attach_discriminator(whitebox_results[str(path)], report)
                    else:
                        candidate_selection = _cli_candidate_selection(
                            path,
                            report,
                            args,
                            content_model_sha256=detector.observed_base_model_sha256.get("content_v2"),
                        )
                        result = engine.score(
                            path,
                            audio_path,
                            candidate_interval_selection=candidate_selection,
                        )
                        result.setdefault("candidate_interval_selection", candidate_selection)
                        result["enabled"] = True
                        result["used_for_final_verdict"] = False
                        result["audio_source"] = "--audio" if args.audio else "AudioFilename"
                        attach_discriminator(result, report)
                        whitebox_results[str(path)] = result
            except (OSError, BeatmapParseError, ValueError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
    finally:
        if engine is not None:
            engine.close()

    payloads = []
    for report in reports:
        payload = report.to_dict(include_features=not args.no_features)
        if str(report.path) in whitebox_results:
            payload.setdefault("analysis_channels", {})["whitebox"] = whitebox_results[str(report.path)]
        payloads.append(payload)
    if args.jsonl:
        for payload in payloads:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        for error in errors:
            print(json.dumps(error, ensure_ascii=False, sort_keys=True))
    elif args.json:
        print(json.dumps({"reports": payloads, "errors": errors}, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        for report in reports:
            if args.compact:
                print(f"{report.verdict.value:22} {report.evidence_score:.4f}  {report.path}")
            else:
                _print_detailed(report, whitebox_results.get(str(report.path)))
        for error in errors:
            print(f"error                  {error['path']}: {error['error']}", file=sys.stderr)

    return 2 if any(report.verdict == Verdict.HIGH_CONFIDENCE_AI for report in reports) else (1 if errors else 0)


if __name__ == "__main__":
    raise SystemExit(main())
