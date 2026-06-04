from pathlib import Path
import argparse
import json
import math

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


KEY_NAMES = [
    "C", "C#/Db", "D", "D#/Eb", "E", "F",
    "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"
]

KEY_TO_INDEX = {k: i for i, k in enumerate(KEY_NAMES)}
INDEX_TO_KEY = {i: k for i, k in enumerate(KEY_NAMES)}

FIFTHS = [
    "C", "G", "D", "A", "E", "B",
    "F#/Gb", "C#/Db", "G#/Ab", "D#/Eb", "A#/Bb", "F"
]

RELATIVE_MINOR = {
    "C": "A",
    "C#/Db": "A#/Bb",
    "D": "B",
    "D#/Eb": "C",
    "E": "C#/Db",
    "F": "D",
    "F#/Gb": "D#/Eb",
    "G": "E",
    "G#/Ab": "F",
    "A": "F#/Gb",
    "A#/Bb": "G",
    "B": "G#/Ab",
}

RELATIVE_MAJOR = {v: k for k, v in RELATIVE_MINOR.items()}


class RewardModel(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def cosine_sim(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def get_vector(row, prefix, n):
    values = []

    for i in range(n):
        col = f"{prefix}_{i}"

        if col not in row:
            raise KeyError(f"Missing required feature column: {col}")

        values.append(float(row[col]))

    return np.asarray(values, dtype=float)


def shift_chroma(chroma, semitones):
    """
    Approximate pitch shifting symbolically by rotating the chroma vector.
    Positive semitones means pitch up.
    """
    semitones = int(semitones)
    return np.roll(chroma, semitones)


def key_index(key):
    return KEY_TO_INDEX.get(str(key))


def signed_shift(original_key, realized_key):
    src = key_index(original_key)
    tgt = key_index(realized_key)

    if src is None or tgt is None:
        return None

    raw = (tgt - src) % 12

    if raw > 6:
        raw -= 12

    return int(raw)


def raw_interval(source_key, target_key):
    src = key_index(source_key)
    tgt = key_index(target_key)

    if src is None or tgt is None:
        return None

    return int((tgt - src) % 12)


def signed_interval(source_key, target_key):
    raw = raw_interval(source_key, target_key)

    if raw is None:
        return None

    if raw > 6:
        raw -= 12

    return int(raw)


def circle_distance(source_key, target_key):
    if source_key not in FIFTHS or target_key not in FIFTHS:
        return 6

    a = FIFTHS.index(source_key)
    b = FIFTHS.index(target_key)

    dist = abs(a - b)
    return int(min(dist, 12 - dist))


def interval_name(raw):
    names = {
        0: "unison",
        1: "minor_second",
        2: "major_second",
        3: "minor_third",
        4: "major_third",
        5: "perfect_fourth",
        6: "tritone",
        7: "perfect_fifth",
        8: "minor_sixth",
        9: "major_sixth",
        10: "minor_seventh",
        11: "major_seventh",
    }

    return names.get(raw, "unknown")


def classify_harmonic_category(source_key, source_mode, target_key, target_mode):
    raw = raw_interval(source_key, target_key)

    if raw is None:
        raw = 0

    signed = signed_interval(source_key, target_key)

    if signed is None:
        signed = 0

    fifth_dist = circle_distance(source_key, target_key)

    same_tonic = int(source_key == target_key)
    same_mode = int(source_mode == target_mode)
    same_key_mode = int(same_tonic and same_mode)
    parallel_mode = int(same_tonic and not same_mode)

    relative_major_to_minor = int(
        source_mode == "major"
        and target_mode == "minor"
        and RELATIVE_MINOR.get(source_key) == target_key
    )

    relative_minor_to_major = int(
        source_mode == "minor"
        and target_mode == "major"
        and RELATIVE_MAJOR.get(source_key) == target_key
    )

    relative_major_minor = int(relative_major_to_minor or relative_minor_to_major)

    perfect_fifth_relation = int(raw in {5, 7})
    circle_neighbor = int(fifth_dist == 1)
    near_circle = int(fifth_dist == 2)
    half_step_relation = int(raw in {1, 11})
    whole_step_relation = int(raw in {2, 10})
    tritone_relation = int(raw == 6)
    chromatic_mediant_candidate = int(raw in {3, 4, 8, 9} and source_mode == target_mode)

    if same_key_mode:
        category = "same_key"
    elif relative_major_to_minor:
        category = "relative_major_to_minor"
    elif relative_minor_to_major:
        category = "relative_minor_to_major"
    elif parallel_mode:
        category = "parallel_mode"
    elif circle_neighbor:
        category = "circle_neighbor"
    elif near_circle:
        category = "near_circle"
    elif chromatic_mediant_candidate:
        category = "chromatic_mediant"
    elif half_step_relation:
        category = "half_step_shift"
    elif tritone_relation:
        category = "tritone"
    elif whole_step_relation:
        category = "whole_step_shift"
    else:
        category = "distant_other"

    return {
        "raw_interval_semitones": raw,
        "signed_interval_semitones": signed,
        "abs_interval_semitones": abs(signed),
        "interval_name": interval_name(raw),
        "circle_of_fifths_distance": fifth_dist,
        "same_tonic": same_tonic,
        "same_mode": same_mode,
        "same_key_mode": same_key_mode,
        "parallel_mode": parallel_mode,
        "relative_major_to_minor": relative_major_to_minor,
        "relative_minor_to_major": relative_minor_to_major,
        "relative_major_minor": relative_major_minor,
        "perfect_fifth_relation": perfect_fifth_relation,
        "circle_neighbor": circle_neighbor,
        "near_circle": near_circle,
        "half_step_relation": half_step_relation,
        "whole_step_relation": whole_step_relation,
        "tritone_relation": tritone_relation,
        "chromatic_mediant_candidate": chromatic_mediant_candidate,
        "harmonic_category": category,
    }


def theory_prior_score(harmonic):
    prior_by_category = {
        "same_key": 0.95,
        "relative_minor_to_major": 0.92,
        "relative_major_to_minor": 0.88,
        "circle_neighbor": 0.84,
        "parallel_mode": 0.78,
        "near_circle": 0.70,
        "chromatic_mediant": 0.64,
        "whole_step_shift": 0.52,
        "half_step_shift": 0.38,
        "tritone": 0.30,
        "distant_other": 0.25,
    }

    return float(prior_by_category.get(harmonic["harmonic_category"], 0.25))


def useful_destination_keys(current_key, current_mode):
    """
    Candidate realized keys for the next section.

    This is intentionally not all 12 keys. It includes:
      - same key
      - circle-of-fifths neighbors
      - relative major/minor space
      - a few expressive movements for exploration
    """
    if current_key not in KEY_TO_INDEX:
        return []

    idx = KEY_TO_INDEX[current_key]
    candidates = set()

    # Same key.
    candidates.add(current_key)

    # Dominant/subdominant / circle neighbors.
    for shift in [7, -7, 5, -5]:
        candidates.add(INDEX_TO_KEY[(idx + shift) % 12])

    # Whole-step and mediant-ish options.
    for shift in [2, -2, 3, 4, -3, -4]:
        candidates.add(INDEX_TO_KEY[(idx + shift) % 12])

    # Relative major/minor space.
    if current_mode == "major" and current_key in RELATIVE_MINOR:
        candidates.add(RELATIVE_MINOR[current_key])
    elif current_mode == "minor" and current_key in RELATIVE_MAJOR:
        candidates.add(RELATIVE_MAJOR[current_key])

    return sorted(candidates)


def load_model(model_dir):
    model_dir = Path(model_dir)

    with open(model_dir / "reward_model_config.json", "r") as f:
        config = json.load(f)

    scaler = joblib.load(model_dir / "reward_scaler.pkl")

    model = RewardModel(config["input_dim"])
    model.load_state_dict(torch.load(model_dir / "reward_model.pt", map_location="cpu"))
    model.eval()

    return model, scaler, config


def build_feature_row(edge, config):
    numeric_cols = config["feature_cols_numeric"]
    categorical_cols = config["categorical_cols"]
    expanded_cols = config["expanded_feature_cols"]

    row_df = pd.DataFrame([edge])

    missing_numeric = [c for c in numeric_cols if c not in row_df.columns]
    missing_categorical = [c for c in categorical_cols if c not in row_df.columns]

    if missing_numeric:
        raise KeyError(f"Planning edge missing numeric columns: {missing_numeric}")

    if missing_categorical:
        raise KeyError(f"Planning edge missing categorical columns: {missing_categorical}")

    numeric = row_df[numeric_cols].astype(float)

    categorical = pd.get_dummies(
        row_df[categorical_cols].fillna("unknown").astype(str),
        prefix=categorical_cols,
    )

    feat = pd.concat([numeric, categorical], axis=1)

    for col in expanded_cols:
        if col not in feat.columns:
            feat[col] = 0.0

    feat = feat[expanded_cols]

    return feat.iloc[0].astype(float).values


def score_edges_with_model(edges, model, scaler, config):
    xs = []

    for edge in edges:
        xs.append(build_feature_row(edge, config))

    x = np.asarray(xs, dtype=np.float32)
    x = scaler.transform(x).astype(np.float32)

    with torch.no_grad():
        scores = model(torch.tensor(x)).numpy()

    return scores


def make_edge(source, target, destination_key, args):
    """
    Build one planning-time candidate edge.

    It represents:
      source section in its current key/mode
      -> target section realized in destination_key

    No audio rendering happens here. Pitch shift is symbolic during planning
    and applied later only when rendering a preview.
    """
    source_key = source["exit_key"]
    source_mode = source["exit_mode"]

    target_original_key = target["entry_key"]
    target_mode = target["entry_mode"]

    target_shift = signed_shift(target_original_key, destination_key)

    if target_shift is None:
        return None

    if abs(target_shift) > args.max_abs_pitch_shift:
        return None

    harmonic = classify_harmonic_category(
        source_key=source_key,
        source_mode=source_mode,
        target_key=destination_key,
        target_mode=target_mode,
    )

    prior = theory_prior_score(harmonic)

    source_exit_chroma = get_vector(source, "exit_chroma", 12)
    target_entry_chroma = get_vector(target, "entry_chroma", 12)

    # Approximate the target's realized key by rotating its chroma representation.
    target_entry_chroma_shifted = shift_chroma(target_entry_chroma, target_shift)

    chroma_sim_raw = cosine_sim(source_exit_chroma, target_entry_chroma_shifted)

    source_mfcc = get_vector(source, "mfcc_mean", 20)
    target_mfcc = get_vector(target, "mfcc_mean", 20)
    mfcc_sim = cosine_sim(source_mfcc, target_mfcc)

    energy_change = float(target["entry_rms_mean"]) - float(source["exit_rms_mean"])
    energy_abs_diff = abs(energy_change)
    energy_score = 1.0 - min(energy_abs_diff / 0.15, 1.0)

    preserves_original = int(target_shift == 0)
    abs_shift = abs(target_shift)

    # Strong original-key preference, but not a hard rule.
    transposition_prior = math.exp(-args.pitch_shift_decay * abs_shift)

    baseline_score = (
        0.35 * prior
        + 0.25 * transposition_prior
        + 0.20 * chroma_sim_raw
        + 0.10 * energy_score
        + 0.10 * mfcc_sim
    )

    edge = {
        "transition_id": "",
        "source_section_id": source["section_id"],
        "target_section_id": target["section_id"],
        "source_track_id": source["track_id"],
        "target_track_id": target["track_id"],

        "source_exit_key": source_key,
        "source_exit_mode": source_mode,
        "target_entry_key": destination_key,
        "target_entry_mode": target_mode,

        "target_original_key": target_original_key,
        "target_realized_key": destination_key,
        "target_pitch_shift": target_shift,
        "target_abs_pitch_shift": abs_shift,
        "target_preserves_original_key": preserves_original,
        "transposition_prior": transposition_prior,

        # Compatibility with reward model features.
        "source_pitch_shift": 0,
        "source_abs_pitch_shift": 0,
        "source_preserves_original_key": 1,
        "total_abs_pitch_shift": abs_shift,

        **harmonic,

        "theory_prior_score": prior,
        "chroma_sim_raw": chroma_sim_raw,
        "chroma_sim_after_shift": chroma_sim_raw,
        "mfcc_sim": mfcc_sim,
        "energy_change": energy_change,
        "energy_abs_diff": energy_abs_diff,
        "source_exit_chroma_entropy": float(source["exit_chroma_entropy"]),
        "target_entry_chroma_entropy": float(target["entry_chroma_entropy"]),
        "baseline_score": baseline_score,
        "sampling_bucket": "planning_bridge" if not preserves_original else "natural_key",
    }

    return edge


def prepare_features(section_features, boundary_features):
    """
    Merge section-level and boundary-level features.

    These are computed from the user's local audio files, not from FMA metadata.
    """
    df = section_features.merge(
        boundary_features,
        on=["section_id", "track_id", "section_label", "start_time", "end_time"],
        how="inner",
        suffixes=("", "_boundary"),
    )

    return df


def path_uses_track(beam, track_id):
    return track_id in set(beam["tracks"])


def beam_search(features, model, scaler, config, args):
    features_by_section = {
        row["section_id"]: row
        for _, row in features.iterrows()
    }

    all_sections = list(features["section_id"])

    if args.start_section_id:
        if args.start_section_id not in features_by_section:
            raise ValueError(f"start_section_id not found: {args.start_section_id}")
        start_sections = [args.start_section_id]
    else:
        start_sections = all_sections[: args.initial_sections]

    beams = []

    for start_id in start_sections:
        start = features_by_section[start_id]

        beams.append(
            {
                "sections": [start_id],
                "tracks": [start["track_id"]],
                "edges": [],
                "score": 0.0,
                "current_key": start["exit_key"],
                "current_mode": start["exit_mode"],
            }
        )

    transition_counter = 0

    for step in range(args.path_length - 1):
        proposed = []

        for beam in beams:
            source = features_by_section[beam["sections"][-1]].copy()

            # Markovian state update:
            # the current key/mode is whatever the previous transition realized.
            source["exit_key"] = beam["current_key"]
            source["exit_mode"] = beam["current_mode"]

            destination_keys = useful_destination_keys(
                beam["current_key"],
                beam["current_mode"],
            )

            edge_batch = []

            for target_id in all_sections:
                target = features_by_section[target_id]

                if target_id in beam["sections"]:
                    continue

                if not args.allow_repeated_tracks and path_uses_track(beam, target["track_id"]):
                    continue

                for destination_key in destination_keys:
                    edge = make_edge(source, target, destination_key, args)

                    if edge is None:
                        continue

                    edge["transition_id"] = f"PLANEDGE{transition_counter:08d}"
                    transition_counter += 1

                    edge_batch.append(edge)

            if not edge_batch:
                continue

            # Prefilter by baseline score for runtime.
            edge_batch = sorted(
                edge_batch,
                key=lambda e: e["baseline_score"],
                reverse=True,
            )[: args.candidate_edges_per_step]

            rewards = score_edges_with_model(edge_batch, model, scaler, config)

            for edge, reward in zip(edge_batch, rewards):
                edge["reward_score"] = float(reward)

            reward_values = np.asarray([e["reward_score"] for e in edge_batch], dtype=float)

            if reward_values.max() == reward_values.min():
                reward_norms = np.full(len(edge_batch), 0.5)
            else:
                reward_norms = (
                    (reward_values - reward_values.min())
                    / (reward_values.max() - reward_values.min())
                )

            for edge, reward_norm in zip(edge_batch, reward_norms):
                original_bonus = (
                    args.original_key_bonus
                    if edge["target_preserves_original_key"]
                    else 0.0
                )

                shift_penalty = args.pitch_shift_penalty * edge["target_abs_pitch_shift"]

                local_edge_score = (
                    args.reward_weight * float(reward_norm)
                    + args.theory_weight * float(edge["theory_prior_score"])
                    + args.baseline_weight * float(edge["baseline_score"])
                    + original_bonus
                    - shift_penalty
                )

                new_beam = {
                    "sections": beam["sections"] + [edge["target_section_id"]],
                    "tracks": beam["tracks"] + [edge["target_track_id"]],
                    "edges": beam["edges"] + [edge],
                    "score": beam["score"] + local_edge_score,
                    "current_key": edge["target_realized_key"],
                    "current_mode": edge["target_entry_mode"],
                }

                proposed.append(new_beam)

        if not proposed:
            print(f"No proposed expansions at step {step + 1}.")
            break

        beams = sorted(
            proposed,
            key=lambda b: b["score"],
            reverse=True,
        )[: args.beam_size]

        print(f"Step {step + 1}: kept {len(beams)} beams")

    return beams


def save_plans(beams, output_csv, output_json):
    rows = []
    json_out = []

    for plan_idx, beam in enumerate(beams):
        plan_id = f"PLAN{plan_idx:03d}"

        rows.append(
            {
                "plan_id": plan_id,
                "plan_score": beam["score"],
                "step": 0,
                "section_id": beam["sections"][0],
                "track_id": beam["tracks"][0],
                "original_key": "",
                "realized_key": "",
                "pitch_shift": "",
                "incoming_transition_id": "",
                "incoming_from_key": "",
                "incoming_to_key": "",
                "incoming_harmonic_category": "",
                "incoming_interval_name": "",
                "incoming_reward_score": "",
                "target_preserves_original_key": "",
                "target_abs_pitch_shift": "",
                "theory_prior_score": "",
                "baseline_score": "",
            }
        )

        for step, edge in enumerate(beam["edges"], start=1):
            rows.append(
                {
                    "plan_id": plan_id,
                    "plan_score": beam["score"],
                    "step": step,
                    "section_id": edge["target_section_id"],
                    "track_id": edge["target_track_id"],
                    "original_key": edge["target_original_key"],
                    "realized_key": edge["target_realized_key"],
                    "pitch_shift": edge["target_pitch_shift"],
                    "incoming_transition_id": edge["transition_id"],
                    "incoming_from_key": f"{edge['source_exit_key']} {edge['source_exit_mode']}",
                    "incoming_to_key": f"{edge['target_realized_key']} {edge['target_entry_mode']}",
                    "incoming_harmonic_category": edge["harmonic_category"],
                    "incoming_interval_name": edge["interval_name"],
                    "incoming_reward_score": edge["reward_score"],
                    "target_preserves_original_key": edge["target_preserves_original_key"],
                    "target_abs_pitch_shift": edge["target_abs_pitch_shift"],
                    "theory_prior_score": edge["theory_prior_score"],
                    "baseline_score": edge["baseline_score"],
                }
            )

        json_out.append(
            {
                "plan_id": plan_id,
                "plan_score": beam["score"],
                "sections": beam["sections"],
                "tracks": beam["tracks"],
                "edges": beam["edges"],
            }
        )

    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_json = Path(output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w") as f:
        json.dump(json_out, f, indent=2)

    print(f"Saved plans to {out_csv}")
    print(f"Saved JSON to {out_json}")


def print_summary(beams, top_n=5):
    for i, beam in enumerate(beams[:top_n]):
        print("\n" + "=" * 90)
        print(f"PLAN {i:03d} | score={beam['score']:.3f}")
        print("=" * 90)
        print(f"Start: {beam['sections'][0]}")

        for step, edge in enumerate(beam["edges"], start=1):
            shift = int(edge["target_pitch_shift"])

            if shift == 0:
                shift_text = "original key"
            else:
                shift_text = f"shift {shift:+} semitones"

            print(
                f"{step}. {edge['source_section_id']} "
                f"({edge['source_exit_key']} {edge['source_exit_mode']})"
                f" → {edge['target_section_id']} "
                f"as {edge['target_realized_key']} {edge['target_entry_mode']} "
                f"({shift_text})"
            )
            print(
                f"   interval={edge['interval_name']}, "
                f"category={edge['harmonic_category']}, "
                f"reward={edge['reward_score']:.3f}, "
                f"prior={edge['theory_prior_score']:.3f}, "
                f"baseline={edge['baseline_score']:.3f}, "
                f"energy_change={edge['energy_change']:+.4f}, "
                f"mfcc_sim={edge['mfcc_sim']:.3f}, "
                f"chroma_sim={edge['chroma_sim_raw']:.3f}, "
                f"shift_abs={edge['target_abs_pitch_shift']}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--section_features_csv",
        default="data/user_uploads/processed/section_features.csv",
    )
    parser.add_argument(
        "--boundary_features_csv",
        default="data/user_uploads/processed/boundary_features.csv",
    )
    parser.add_argument("--model_dir", default="models")
    parser.add_argument(
        "--output_csv",
        default="data/user_uploads/processed/mashup_plans_with_bridges.csv",
    )
    parser.add_argument(
        "--output_json",
        default="data/user_uploads/processed/mashup_plans_with_bridges.json",
    )

    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--initial_sections", type=int, default=20)
    parser.add_argument("--candidate_edges_per_step", type=int, default=120)

    parser.add_argument("--max_abs_pitch_shift", type=int, default=4)
    parser.add_argument("--pitch_shift_decay", type=float, default=0.35)
    parser.add_argument("--pitch_shift_penalty", type=float, default=0.035)
    parser.add_argument("--original_key_bonus", type=float, default=0.08)

    parser.add_argument("--reward_weight", type=float, default=0.70)
    parser.add_argument("--theory_weight", type=float, default=0.15)
    parser.add_argument("--baseline_weight", type=float, default=0.10)

    parser.add_argument("--start_section_id", default="")
    parser.add_argument("--allow_repeated_tracks", action="store_true")

    args = parser.parse_args()

    section_features = pd.read_csv(args.section_features_csv)
    boundary_features = pd.read_csv(args.boundary_features_csv)

    features = prepare_features(section_features, boundary_features)

    model, scaler, config = load_model(args.model_dir)

    beams = beam_search(features, model, scaler, config, args)

    if not beams:
        raise ValueError("No plans found. Try allowing repeated tracks or reducing path length.")

    print_summary(beams, top_n=5)
    save_plans(beams, args.output_csv, args.output_json)


if __name__ == "__main__":
    main()