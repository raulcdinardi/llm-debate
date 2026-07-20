#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable


ARMS = ("real08b", "real4b", "mock2026071001", "mock2026071002")
MISSING_HIGH = 1e30
MISSING_LOW = -1e30
INVARIANTS = {
    "scorer_gold_pass_rate_min": 1.0,
    "probe_a_internal_renderer_contract_pass_rate_min": 1.0,
    "probe_a_prompt_byte_identity_rate_min": 1.0,
    "probe_a_forbidden_prompt_marker_rate_max": 0.0,
    "probe_a_prefill_loss_mask_violation_count_max": 0,
    "probe_a_target_think_leak_rate_max": 0.02,
    "probe_a_target_cap_hit_rate_max": 0.05,
    "probe_a_target_p95_completion_tokens_max": 120,
    "probe_a_target_json_leak_rate_max": 0.0,
    "probe_a_zero_b_token_identity_rate_min": 1.0,
    "probe_a_zero_b_max_logprob_abs_diff_max": 0.0001,
    "probe_b_prefix_extension_identity_rate_min": 1.0,
    "probe_b_forbidden_appended_context_rate_max": 0.0,
    "probe_b_cap_hit_rate_max": 0.0,
    "probe_b_exact_three_item_rate_min": 0.8,
    "probe_b_matched_stop_rate_min": 0.95,
    "probe_b_bytes_after_sentinel_max": 0,
    "probe_b_think_leak_rate_max": 0.02,
    "probe_b_json_leak_rate_max": 0.0,
    "reduced_parse_success_rate_min": 0.8,
    "reduced_all_rollout_groups_alive_rate_min": 1.0,
    "reduced_r1_think_leak_rate_max": 0.02,
    "reduced_r1_cap_hit_rate_max": 0.0,
    "reduced_r1_json_leak_rate_max": 0.0,
    "reduced_r23_cap_hit_rate_max": 0.0,
    "reduced_r23_exact_three_item_rate_min": 0.8,
    "reduced_r23_matched_stop_rate_min": 0.95,
    "reduced_r23_bytes_after_sentinel_max": 0,
    "reduced_r23_think_leak_rate_max": 0.02,
    "reduced_r23_json_leak_rate_max": 0.0,
    "reduced_reward_band_formula_pass_rate_min": 1.0,
    "phase0_all_arms_completed_rate_min": 1.0,
    "phase0_environment_digest_match_rate_min": 1.0,
    "phase0_adapter_hash_match_rate_min": 1.0,
    "phase0_invalid_verdict_rate_max": 0.02,
    "phase0_observed_emitted_winner_r1_example_count_min": 1,
    "phase0_observed_emitted_loser_r1_example_count_max": 0,
    "phase0_observed_unclassified_r1_example_count_max": 0,
    "phase0_winner_r1_expected_observed_delta_abs_max": 0,
    "phase0_r1_nonzero_advantage_example_count_min": 1,
    "phase0_solution_checkpoint_weight_delta_min": 1e-7,
    "phase0_debate_checkpoint_weight_delta_min": 1e-7,
    "phase0_on_policy_logprob_max_abs_diff_max": 0.001,
    "phase0_vram_headroom_min_mib_min": 8000,
    "phase0_seconds_per_arm_step_max": 180,
    "phase0_archive_local_validation_pass_rate_min": 1.0,
    "manual_raw_review_pass_min": 1,
    "phase0_fable_critique_completed_min": 1,
    "phase0_fable_critique_blocking_issue_count_max": 0,
}
MANUAL_INVARIANTS = {
    "phase0_archive_local_validation_pass_rate_min",
    "manual_raw_review_pass_min",
    "phase0_fable_critique_completed_min",
    "phase0_fable_critique_blocking_issue_count_max",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the bounded four-arm Phase-0 run without authorizing Phase 1.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-image-digest", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def read_optional_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected JSON object row: {path}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def metric_name(invariant_name: str) -> tuple[str, str]:
    if invariant_name.endswith("_min"):
        return invariant_name[:-4], "min"
    if invariant_name.endswith("_max"):
        return invariant_name[:-4], "max"
    raise ValueError(f"Invariant has no min/max suffix: {invariant_name}")


def evaluate_invariants(metrics: dict[str, float], names: Iterable[str]) -> dict[str, Any]:
    results = {}
    for invariant_name in names:
        threshold = INVARIANTS[invariant_name]
        name, direction = metric_name(invariant_name)
        observed = float(metrics.get(name, float("nan")))
        passed = math.isfinite(observed) and (
            observed >= float(threshold) if direction == "min" else observed <= float(threshold)
        )
        results[invariant_name] = {
            "metric": name,
            "observed": observed if math.isfinite(observed) else "missing",
            "operator": ">=" if direction == "min" else "<=",
            "threshold": threshold,
            "pass": passed,
        }
    return results


def max_checkpoint_delta(initial_path: Path, final_path: Path) -> float:
    from safetensors.torch import load_file

    initial = load_file(initial_path, device="cpu")
    final = load_file(final_path, device="cpu")
    if set(initial) != set(final):
        raise RuntimeError(
            f"Adapter tensor keys changed between {initial_path} and {final_path}"
        )
    return max(
        float((final[name].float() - initial[name].float()).abs().max().item())
        for name in initial
    )


def arm_step_seconds(resource_rows: list[dict[str, Any]]) -> float:
    starts = [
        float(row["elapsed_s"])
        for row in resource_rows
        if row.get("event") == "stage_start" and row.get("stage") == "step"
    ]
    ends = [
        float(row["elapsed_s"])
        for row in resource_rows
        if row.get("event") == "stage_end" and row.get("stage") == "step"
    ]
    if starts and ends:
        return max(ends) - min(starts)
    elapsed = [float(row.get("elapsed_s", 0.0)) for row in resource_rows]
    return max(elapsed, default=MISSING_HIGH)


def arm_vram_headroom(resource_rows: list[dict[str, Any]]) -> float:
    values = []
    for row in resource_rows:
        for gpu in row.get("gpus", []):
            values.append(float(gpu["memory_total_mib"]) - float(gpu["memory_used_mib"]))
    return min(values, default=MISSING_LOW)


def select_review_rows(arm: str, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for idx, sample in enumerate(samples):
        verdict = sample.get("verdict")
        a_reward = float(sample.get("trajectory_a", {}).get("task_reward", 0.0))
        b_reward = float(sample.get("trajectory_b", {}).get("task_reward", 0.0))
        objective_preference = "A" if a_reward > b_reward else "B" if b_reward > a_reward else "TIE"
        agreement = verdict == objective_preference if objective_preference != "TIE" else False
        winner_reward = a_reward if verdict == "A" else b_reward if verdict == "B" else min(a_reward, b_reward)
        candidates.append({
            "arm": arm,
            "source_index": idx,
            "winner_objective_reward": winner_reward,
            "objective_preference": objective_preference,
            "judge_objective_agreement": agreement,
            **sample,
        })
    ordered = sorted(candidates, key=lambda row: (row["winner_objective_reward"], row["source_index"]))
    if not ordered:
        return []
    target_indices = [0, len(ordered) // 4, len(ordered) // 2, 3 * len(ordered) // 4, len(ordered) - 1]
    selected = []
    seen = set()
    for idx in target_indices:
        row = ordered[idx]
        if row["source_index"] not in seen:
            selected.append(row)
            seen.add(row["source_index"])
    for desired_agreement in (False, True):
        for row in ordered:
            if len(selected) >= 6:
                break
            if row["source_index"] not in seen and row["judge_objective_agreement"] is desired_agreement:
                selected.append(row)
                seen.add(row["source_index"])
    for row in ordered:
        if len(selected) >= 6:
            break
        if row["source_index"] not in seen:
            selected.append(row)
            seen.add(row["source_index"])
    return selected[:6]


def diagnostic_metrics(run_root: Path) -> dict[str, float]:
    probe_path = run_root / "diagnostic" / "probe_summary.json"
    reduced_path = run_root / "diagnostic" / "phase0_summary.json"
    if not probe_path.is_file():
        return {}
    probe = read_json(probe_path)
    a = probe["probe_a"]
    b = probe["probe_b"]
    c = probe["probe_c"]
    target_cells = [value for name, value in a["cells"].items() if name.startswith("target_prefill__")]
    b_cell = b["cells"][b["chosen_cell"]]
    max_logprob_diff = max(
        float(a["zero_b_identity"]["max_chosen_logprob_abs_diff"]),
        float(a["zero_b_identity"]["max_common_top20_logprob_abs_diff"]),
    )
    metrics = {
        "scorer_gold_pass_rate": float(c["passed"]) / max(1.0, float(c["num_cases"])),
        "probe_a_internal_renderer_contract_pass_rate": float(bool(a["internal_renderer_contract_pass"])),
        "probe_a_prompt_byte_identity_rate": float(a["prompt_byte_identity_rate"]),
        "probe_a_forbidden_prompt_marker_rate": float(a["forbidden_prompt_marker_rate"]),
        "probe_a_prefill_loss_mask_violation_count": float(a["prefill_loss_mask_violation_count"]),
        "probe_a_target_think_leak_rate": max(float(cell["think_leak_rate"]) for cell in target_cells),
        "probe_a_target_cap_hit_rate": max(float(cell["cap_rate"]) for cell in target_cells),
        "probe_a_target_p95_completion_tokens": max(float(cell["tokens_p95"]) for cell in target_cells),
        "probe_a_target_json_leak_rate": max(float(cell["json_leak_rate"]) for cell in target_cells),
        "probe_a_zero_b_token_identity_rate": float(bool(a["zero_b_identity"]["all_token_ids_equal"])),
        "probe_a_zero_b_max_logprob_abs_diff": max_logprob_diff,
        "probe_b_prefix_extension_identity_rate": float(b["prefix_extension_identity_rate"]),
        "probe_b_forbidden_appended_context_rate": float(b["forbidden_appended_context_rate"]),
        "probe_b_cap_hit_rate": float(b_cell["cap_rate"]),
        "probe_b_exact_three_item_rate": float(b_cell["exact_three_rate"]),
        "probe_b_matched_stop_rate": float(b_cell["matched_stop_rate"]),
        "probe_b_bytes_after_sentinel": 0.0 if float(b_cell["zero_post_sentinel_rate"]) == 1.0 else 1.0,
        "probe_b_think_leak_rate": float(b_cell["think_leak_rate"]),
        "probe_b_json_leak_rate": float(b_cell["json_leak_rate"]),
    }
    if not reduced_path.is_file():
        return metrics
    reduced = read_json(reduced_path)
    conditions = reduced["conditions"]
    r23 = [condition[round_name] for condition in conditions for round_name in ("r2", "r3")]
    metrics.update({
        "reduced_parse_success_rate": min(float(condition["parse_rate"]) for condition in conditions),
        "reduced_all_rollout_groups_alive_rate": min(float(condition["groups_alive_fraction"]) for condition in conditions),
        "reduced_r1_think_leak_rate": max(float(condition["r1"]["think_leak_rate"]) for condition in conditions),
        "reduced_r1_cap_hit_rate": max(float(condition["r1"]["cap_rate"]) for condition in conditions),
        "reduced_r1_json_leak_rate": max(float(condition["r1"]["json_leak_rate"]) for condition in conditions),
        "reduced_r23_cap_hit_rate": max(float(value["cap_rate"]) for value in r23),
        "reduced_r23_exact_three_item_rate": min(float(value["exact_three_rate"]) for value in r23),
        "reduced_r23_matched_stop_rate": min(float(value["matched_stop_rate"]) for value in r23),
        "reduced_r23_bytes_after_sentinel": 0.0 if all(float(value["zero_post_sentinel_rate"]) == 1.0 for value in r23) else 1.0,
        "reduced_r23_think_leak_rate": max(float(value["think_leak_rate"]) for value in r23),
        "reduced_r23_json_leak_rate": max(float(value["json_leak_rate"]) for value in r23),
        "reduced_reward_band_formula_pass_rate": sum(bool(value["pass"]) for value in reduced["per_condition_gates"]) / max(1, len(reduced["per_condition_gates"])),
    })
    return metrics


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = read_json(run_root / "preparation" / "provenance.json")
    metrics = diagnostic_metrics(run_root)
    arm_details = {}
    review_rows = []
    completed = 0
    invalid_count = 0
    debate_count = 0
    winner_count = 0
    loser_count = 0
    unclassified_count = 0
    expected_delta_abs = 0
    nonzero_advantage_count = 0
    on_policy_max = 0.0
    solution_deltas = []
    debate_deltas = []
    headrooms = []
    step_seconds = []

    initial_solution = Path(provenance["adapters"]["solution_zero_b"]["path"]) / "adapter_model.safetensors"
    initial_debate = Path(provenance["adapters"]["debate"]["path"]) / "adapter_model.safetensors"
    for arm in ARMS:
        arm_dir = run_root / "arms" / arm
        records = read_jsonl(arm_dir / "step_records.jsonl")
        summary = read_optional_json(arm_dir / "summary.json")
        resource_rows = read_jsonl(arm_dir / "resource_usage.jsonl")
        complete = len(records) == 1 and int(summary.get("num_steps_completed", 0)) == 1
        if complete:
            completed += 1
        detail: dict[str, Any] = {"complete": complete, "step_records": len(records)}
        if records:
            record = records[0]
            projection = record.get("r1_projection", {})
            n_debates = int(record.get("num_debates", projection.get("valid_verdict_count", 0) + projection.get("invalid_verdict_count", 0)))
            debate_count += n_debates
            invalid_count += int(projection.get("invalid_verdict_count", n_debates))
            winner_count += int(projection.get("winner_r1_example_count", 0))
            loser_count += int(projection.get("loser_r1_example_count", 0))
            unclassified_count += int(projection.get("unclassified_r1_example_count", 0))
            expected_delta_abs = max(
                expected_delta_abs,
                abs(int(projection.get("emitted_r1_example_count_delta", n_debates))),
                abs(int(projection.get("winner_r1_example_count_delta", n_debates))),
            )
            nonzero_advantage_count += int(projection.get("nonzero_advantage_r1_example_count", 0))
            for adapter_metrics in record.get("train_metrics", {}).values():
                on_policy_max = max(
                    on_policy_max,
                    float(adapter_metrics.get("on_policy_logprob_max_abs_diff", float("inf"))),
                )
            review_rows.extend(select_review_rows(arm, record.get("sample_records", [])))
            manifest = read_optional_json(arm_dir / "manifest.json")
            adapter_dirs = manifest.get("adapter_dirs", {})
            if "solution" in adapter_dirs and "debate" in adapter_dirs:
                solution_delta = max_checkpoint_delta(
                    initial_solution, Path(adapter_dirs["solution"]) / "adapter_model.safetensors"
                )
                debate_delta = max_checkpoint_delta(
                    initial_debate, Path(adapter_dirs["debate"]) / "adapter_model.safetensors"
                )
                solution_deltas.append(solution_delta)
                debate_deltas.append(debate_delta)
                detail["solution_checkpoint_max_abs_delta"] = solution_delta
                detail["debate_checkpoint_max_abs_delta"] = debate_delta
        if resource_rows:
            headroom = arm_vram_headroom(resource_rows)
            seconds = arm_step_seconds(resource_rows)
            headrooms.append(headroom)
            step_seconds.append(seconds)
            detail["vram_headroom_min_mib"] = headroom
            detail["step_seconds"] = seconds
        arm_details[arm] = detail

    adapter_records = provenance["adapters"]
    adapter_hash_matches = [
        adapter_records["debate"]["adapter_model_sha256"] == "b97cea47bcf0ec8ab501dcc14b234b7ca28a84237e4a09094ee9ff04e2bef209",
        adapter_records["judge_08b"]["adapter_model_sha256"] == "6e284aaa96ecc21749be5f40bddbd4b54247d9ea2f36c0e562e5cf8a4d3f5f89",
        adapter_records["judge_4b"]["adapter_model_sha256"] == "357996688954bd59a7c28eeb0632f12aee247b776111ed651cc1b904f18a7582",
    ]
    metrics.update({
        "phase0_all_arms_completed_rate": completed / len(ARMS),
        "phase0_environment_digest_match_rate": float(
            provenance["source"]["commit"] == args.expected_source_commit
            and provenance["image"]["digest"] == args.expected_image_digest
        ),
        "phase0_adapter_hash_match_rate": sum(adapter_hash_matches) / len(adapter_hash_matches),
        "phase0_invalid_verdict_rate": invalid_count / max(1, debate_count),
        "phase0_observed_emitted_winner_r1_example_count": float(winner_count),
        "phase0_observed_emitted_loser_r1_example_count": float(loser_count),
        "phase0_observed_unclassified_r1_example_count": float(unclassified_count),
        "phase0_winner_r1_expected_observed_delta_abs": float(expected_delta_abs),
        "phase0_r1_nonzero_advantage_example_count": float(nonzero_advantage_count),
        "phase0_solution_checkpoint_weight_delta": min(solution_deltas, default=0.0),
        "phase0_debate_checkpoint_weight_delta": min(debate_deltas, default=0.0),
        "phase0_on_policy_logprob_max_abs_diff": on_policy_max if debate_count > 0 else MISSING_HIGH,
        "phase0_vram_headroom_min_mib": min(headrooms, default=MISSING_LOW),
        "phase0_seconds_per_arm_step": max(step_seconds, default=MISSING_HIGH),
    })

    archive_validation = read_optional_json(run_root / "archive_validation.json")
    manual_review = read_optional_json(run_root / "manual_review.json")
    fable = read_optional_json(run_root / "fable_critique.json")
    metrics.update({
        "phase0_archive_local_validation_pass_rate": float(archive_validation.get("pass_rate", 0.0)),
        "manual_raw_review_pass": float(bool(manual_review.get("pass", False))),
        "phase0_fable_critique_completed": float(bool(fable.get("completed", False))),
        "phase0_fable_critique_blocking_issue_count": float(fable.get("blocking_issue_count", MISSING_HIGH)),
    })

    automatic_names = [name for name in INVARIANTS if name not in MANUAL_INVARIANTS]
    automatic_results = evaluate_invariants(metrics, automatic_names)
    manual_results = evaluate_invariants(metrics, MANUAL_INVARIANTS)
    automatic_pass = all(value["pass"] for value in automatic_results.values())
    lifecycle_complete = automatic_pass and all(value["pass"] for value in manual_results.values())
    write_jsonl(output_path.parent / "raw_review_24.jsonl", review_rows[:24])
    result = {
        "schema": "cw_judge_signal_phase0_preflight_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "invariants": INVARIANTS,
        "automatic_gate": {"pass": automatic_pass, "results": automatic_results},
        "post_collection_gate": {"pass": lifecycle_complete, "results": manual_results},
        "arms": arm_details,
        "raw_review_count": min(24, len(review_rows)),
        "manual_raw_review_required": True,
        "local_archive_validation_required": True,
        "fable_critique_required_after_local_validation": True,
        "phase0_lifecycle_complete": lifecycle_complete,
        "phase1_forbidden": True,
        "phase1_release_authority": "later explicit human decision only",
    }
    write_json(output_path, result)
    marker = (
        output_path.parent / "PHASE0_QUANTITATIVE_PASS_RAW_REVIEW_REQUIRED"
        if automatic_pass
        else output_path.parent / "PHASE0_QUANTITATIVE_REJECT"
    )
    marker.touch()
    print(json.dumps({
        "event": "phase0_preflight_finalized",
        "automatic_pass": automatic_pass,
        "lifecycle_complete": lifecycle_complete,
        "phase1_forbidden": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
