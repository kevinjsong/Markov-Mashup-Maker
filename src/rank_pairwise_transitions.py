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


class RewardModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
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
    return np.roll(chroma, int(semitones))


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
    if current_key not in KEY_TO_INDEX:
        return []

    idx = KEY_TO_INDEX[current_key]
    candidates = set()

    # Same key.
    candidates.add(current_key)

    # Smooth/familiar options.
    for shift in [7, -7, 5, -5]:
        candidates.add(INDEX_TO_KEY[(idx + shift) % 12])

    # More adventurous options.
    for shift in [2, -2, 3, 4, -3, -4, 6]:
        candidates.add(INDEX_TO_KEY[(idx + shift) % 12])

    # Relative major/minor.
    if current_mode == "major" and current_key in RELATIVE_MINOR:
        candidates.add(RELATIVE_MINOR[current_key])
    elif current_mode == "minor" and current_key in RELATIVE_MAJOR:
        candidates.add(RELATIVE_MAJOR[current_key])

    return sorted(candidates)


def discovery_bonus(harmonic_category, discovery_goal):
    """
    discovery_goal:
      smooth   -> prioritize seamless / compatible transitions
      creative -> prioritize more surprising intervals that still score well
    """
    smooth_bonus = {
        "same_key": 0.30,
        "circle_neighbor": 0.24,
        "relative_minor_to_major": 0.22,
        "relative_major_to_minor": 0.22,
        "parallel_mode": 0.08,
        "near_circle": 0.06,
        "chromatic_mediant": -0.03,
        "whole_step_shift": -0.04,
        "half_step_shift": -0.12,
        "tritone": -0.18,
        "distant_other": -0.20,
    }

    creative_bonus = {
        "same_key": -0.25,
        "circle_neighbor": -0.05,
        "relative_minor_to_major": -0.02,
        "relative_major_to_minor": -0.02,
        "parallel_mode": 0.12,
        "near_circle": 0.14,
        "chromatic_mediant": 0.30,
        "whole_step_shift": 0.20,
        "half_step_shift": 0.22,
        "tritone": 0.24,
        "distant_other": 0.10,
    }

    if discovery_goal == "creative":
        return float(creative_bonus.get(harmonic_category, 0.0))

    return float(smooth_bonus.get(harmonic_category, 0.0))


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
    xs = [build_feature_row(edge, config) for edge in edges]

    x = np.asarray(xs, dtype=np.float32)
    x = scaler.transform(x).astype(np.float32)

    with torch.no_grad():
        scores = model(torch.tensor(x)).numpy()

    return scores


def make_edge(source, target, destination_key, args):
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

        # Compatibility with reward model feature schema.
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
        "sampling_bucket": "pairwise_discovery",
    }

    return edge


def prepare_features(section_features, boundary_features):
    return section_features.merge(
        boundary_features,
        on=["section_id", "track_id", "section_label", "start_time", "end_time"],
        how="inner",
        suffixes=("", "_boundary"),
    )


def attach_track_durations(features, tracks_csv):
    tracks_csv = Path(tracks_csv)

    if not tracks_csv.exists():
        return features

    tracks = pd.read_csv(tracks_csv)

    if "track_id" not in tracks.columns or "duration" not in tracks.columns:
        return features

    durations = tracks[["track_id", "duration"]].rename(
        columns={"duration": "track_duration"}
    )

    return features.merge(durations, on="track_id", how="left")


def is_final_source_section(row, tolerance_sec=3.0):
    """
    True if this section ends near the end of its original track.

    In a cappella / split-song mode, we usually do not want final/outro sections
    to be used as source sections, because the transition should happen before
    the song fully ends.
    """
    if "track_duration" not in row:
        return False

    if pd.isna(row["track_duration"]):
        return False

    return float(row["end_time"]) >= float(row["track_duration"]) - float(tolerance_sec)


def normalize_values(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    if values.max() == values.min():
        return np.full(len(values), 0.5)

    return (values - values.min()) / (values.max() - values.min())


def rank_pairwise_transitions(features, model, scaler, config, args):
    all_sections = list(features["section_id"])
    rows_by_section = {
        row["section_id"]: row
        for _, row in features.iterrows()
    }

    edges = []
    edge_counter = 0
    skipped_final_sources = 0

    for source_id in all_sections:
        source = rows_by_section[source_id]

        if args.exclude_final_source_sections and is_final_source_section(
            source,
            tolerance_sec=args.final_section_tolerance_sec,
        ):
            skipped_final_sources += 1
            continue

        destination_keys = useful_destination_keys(source["exit_key"], source["exit_mode"])

        for target_id in all_sections:
            if source_id == target_id:
                continue

            target = rows_by_section[target_id]

            if not args.allow_same_track and source["track_id"] == target["track_id"]:
                continue

            for destination_key in destination_keys:
                edge = make_edge(source, target, destination_key, args)

                if edge is None:
                    continue

                edge["transition_id"] = f"PAIRWISE{edge_counter:08d}"
                edge["discovery_goal"] = args.discovery_goal
                edge["discovery_bonus"] = discovery_bonus(
                    edge["harmonic_category"],
                    args.discovery_goal,
                )
                edge_counter += 1
                edges.append(edge)

    if not edges:
        raise ValueError("No pairwise transitions generated.")

    print(f"Generated {len(edges)} candidate pairwise transitions.")
    if args.exclude_final_source_sections:
        print(f"Skipped {skipped_final_sources} final source sections.")

    # Runtime prefilter. For creative mode, include discovery bonus in the prefilter
    # so adventurous transitions are not discarded before neural scoring.
    if args.discovery_goal == "creative":
        edges = sorted(
            edges,
            key=lambda e: e["baseline_score"] + args.discovery_weight * e["discovery_bonus"],
            reverse=True,
        )
    else:
        edges = sorted(edges, key=lambda e: e["baseline_score"], reverse=True)

    edges = edges[: args.max_candidates_to_score]

    reward_scores = score_edges_with_model(edges, model, scaler, config)

    for edge, reward in zip(edges, reward_scores):
        edge["reward_score"] = float(reward)

    reward_norms = normalize_values([e["reward_score"] for e in edges])

    for edge, reward_norm in zip(edges, reward_norms):
        original_bonus = args.original_key_bonus if edge["target_preserves_original_key"] else 0.0
        shift_penalty = args.pitch_shift_penalty * edge["target_abs_pitch_shift"]

        edge["final_edge_score"] = (
            args.reward_weight * float(reward_norm)
            + args.theory_weight * float(edge["theory_prior_score"])
            + args.baseline_weight * float(edge["baseline_score"])
            + original_bonus
            - shift_penalty
            + args.discovery_weight * edge["discovery_bonus"]
        )

    ranked = pd.DataFrame(edges)
    ranked = ranked.sort_values("final_edge_score", ascending=False).head(args.top_n).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    return ranked


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
    parser.add_argument(
        "--tracks_csv",
        default="data/user_uploads/processed/tracks.csv",
    )
    parser.add_argument("--model_dir", default="models")
    parser.add_argument(
        "--output_csv",
        default="data/user_uploads/processed/best_pairwise_transitions.csv",
    )

    parser.add_argument(
        "--discovery_goal",
        choices=["smooth", "creative"],
        default="smooth",
    )

    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--max_candidates_to_score", type=int, default=5000)

    parser.add_argument("--max_abs_pitch_shift", type=int, default=4)
    parser.add_argument("--pitch_shift_decay", type=float, default=0.35)
    parser.add_argument("--pitch_shift_penalty", type=float, default=0.035)
    parser.add_argument("--original_key_bonus", type=float, default=0.08)

    parser.add_argument("--reward_weight", type=float, default=0.65)
    parser.add_argument("--theory_weight", type=float, default=0.12)
    parser.add_argument("--baseline_weight", type=float, default=0.08)
    parser.add_argument("--discovery_weight", type=float, default=0.35)

    parser.add_argument("--allow_same_track", action="store_true")

    parser.add_argument(
        "--exclude_final_source_sections",
        action="store_true",
        help=(
            "For a cappella / split-song mode: prevent final/outro sections "
            "from being used as source sections in pairwise transition discovery."
        ),
    )

    parser.add_argument(
        "--final_section_tolerance_sec",
        type=float,
        default=3.0,
        help="A section is final if it ends within this many seconds of the track duration.",
    )

    args = parser.parse_args()

    section_features = pd.read_csv(args.section_features_csv)
    boundary_features = pd.read_csv(args.boundary_features_csv)

    features = prepare_features(section_features, boundary_features)
    features = attach_track_durations(features, args.tracks_csv)

    model, scaler, config = load_model(args.model_dir)

    ranked = rank_pairwise_transitions(features, model, scaler, config, args)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_path, index=False)

    print(f"Saved ranked pairwise transitions to {out_path}")

    preview_cols = [
        "rank",
        "source_section_id",
        "target_section_id",
        "source_exit_key",
        "source_exit_mode",
        "target_original_key",
        "target_realized_key",
        "target_pitch_shift",
        "interval_name",
        "harmonic_category",
        "reward_score",
        "baseline_score",
        "discovery_bonus",
        "final_edge_score",
    ]

    existing = [c for c in preview_cols if c in ranked.columns]
    print(ranked[existing].head(25).to_string(index=False))

    print("\nHarmonic category counts:")
    print(ranked["harmonic_category"].value_counts())


if __name__ == "__main__":
    main()