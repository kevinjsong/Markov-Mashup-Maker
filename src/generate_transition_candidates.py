from pathlib import Path
import argparse
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


KEY_NAMES = [
    "C", "C#/Db", "D", "D#/Eb", "E", "F",
    "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"
]

KEY_TO_INDEX = {k: i for i, k in enumerate(KEY_NAMES)}

# Circle of fifths order.
FIFTHS = ["C", "G", "D", "A", "E", "B", "F#/Gb", "C#/Db", "G#/Ab", "D#/Eb", "A#/Bb", "F"]

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


def key_index(key):
    return KEY_TO_INDEX.get(str(key))


def raw_interval_semitones(source_key, target_key):
    """
    Upward pitch-class distance from source to target in [0, 11].
    """
    src = key_index(source_key)
    tgt = key_index(target_key)

    if src is None or tgt is None:
        return None

    return int((tgt - src) % 12)


def signed_interval_semitones(source_key, target_key):
    """
    Shortest signed pitch-class distance from source to target in [-6, +6].
    """
    raw = raw_interval_semitones(source_key, target_key)

    if raw is None:
        return None

    if raw > 6:
        raw -= 12

    return int(raw)


def circle_distance(source_key, target_key):
    if source_key not in FIFTHS or target_key not in FIFTHS:
        return None

    a = FIFTHS.index(source_key)
    b = FIFTHS.index(target_key)

    dist = abs(a - b)
    return int(min(dist, 12 - dist))


def cos_sim(a, b):
    a = np.asarray(a, dtype=float).reshape(1, -1)
    b = np.asarray(b, dtype=float).reshape(1, -1)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    return float(cosine_similarity(a, b)[0, 0])


def interval_name(raw_interval):
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
    return names.get(raw_interval, "unknown")


def classify_harmonic_category(source_key, source_mode, target_key, target_mode):
    """
    Theory-informed category for source exit key/mode -> target entry key/mode.

    This is a structured prior, not a hard truth.
    """
    raw = raw_interval_semitones(source_key, target_key)
    signed = signed_interval_semitones(source_key, target_key)
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

    # In key-center terms, a fifth/fourth relation is often closely related.
    perfect_fifth_relation = int(raw in {5, 7})
    circle_neighbor = int(fifth_dist == 1) if fifth_dist is not None else 0
    near_circle = int(fifth_dist == 2) if fifth_dist is not None else 0

    half_step_relation = int(raw in {1, 11})
    whole_step_relation = int(raw in {2, 10})
    tritone_relation = int(raw == 6)

    # Chromatic mediants: roots a major/minor third apart, usually same mode.
    chromatic_mediant_candidate = int(raw in {3, 4, 8, 9} and source_mode == target_mode)

    # Category priority.
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
        "abs_interval_semitones": abs(signed) if signed is not None else None,
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


def theory_prior_score(features):
    """
    Theory-informed prior score.

    Important:
      This is not "objective goodness."
      It is a structured prior used to seed search and weak labels.

    Human feedback can override this.
    """
    category = features["harmonic_category"]

    prior_by_category = {
        # Safest / strongest conventional relations
        "same_key": 0.95,
        "relative_minor_to_major": 0.92,
        "relative_major_to_minor": 0.88,
        "circle_neighbor": 0.84,
        "parallel_mode": 0.78,

        # Still plausible but more context-dependent
        "near_circle": 0.70,
        "chromatic_mediant": 0.64,
        "whole_step_shift": 0.52,

        # Riskier; include for exploration, not preferred by default
        "half_step_shift": 0.38,
        "tritone": 0.30,
        "distant_other": 0.25,
    }

    return float(prior_by_category.get(category, 0.25))


def sampling_bucket(row):
    """
    Used later for balanced / epsilon-greedy feedback sampling.
    """
    category = row["harmonic_category"]

    if category in {"same_key", "relative_minor_to_major", "relative_major_to_minor", "circle_neighbor"}:
        return "safe_theory"
    if category in {"parallel_mode", "near_circle", "chromatic_mediant", "whole_step_shift"}:
        return "interesting_theory"
    return "risky_explore"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section_features_csv",
        default="data/training_processed/section_features.csv",
    )
    parser.add_argument(
        "--boundary_features_csv",
        default="data/training_processed/boundary_features.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="data/training_processed/transition_candidates.csv",
    )
    parser.add_argument("--max_candidates", type=int, default=30000)
    parser.add_argument("--same_track_allowed", action="store_true")
    parser.add_argument("--top_k_per_source", type=int, default=40)
    parser.add_argument("--epsilon", type=float, default=0.20)

    args = parser.parse_args()

    sf = pd.read_csv(args.section_features_csv)
    bf = pd.read_csv(args.boundary_features_csv)

    features = sf.merge(
        bf,
        on=["section_id", "track_id", "section_label", "start_time", "end_time"],
        how="inner",
        suffixes=("", "_boundary"),
    )

    entry_chroma_cols = [f"entry_chroma_{i}" for i in range(12)]
    exit_chroma_cols = [f"exit_chroma_{i}" for i in range(12)]
    mfcc_cols = [f"mfcc_mean_{i}" for i in range(20)]

    required = [
        "section_id",
        "track_id",
        "entry_key",
        "entry_mode",
        "exit_key",
        "exit_mode",
        "entry_rms_mean",
        "exit_rms_mean",
        "entry_chroma_entropy",
        "exit_chroma_entropy",
    ]

    missing = [c for c in required if c not in features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    rows_by_source = {}
    transition_counter = 0

    pairs = itertools.product(features.iterrows(), features.iterrows())

    for (_, source), (_, target) in tqdm(pairs, total=len(features) * len(features)):
        if not args.same_track_allowed and source["track_id"] == target["track_id"]:
            continue

        source_key = source["exit_key"]
        source_mode = source["exit_mode"]
        target_key = target["entry_key"]
        target_mode = target["entry_mode"]

        harmonic = classify_harmonic_category(
            source_key=source_key,
            source_mode=source_mode,
            target_key=target_key,
            target_mode=target_mode,
        )

        prior = theory_prior_score(harmonic)

        source_chroma = source[exit_chroma_cols].values.astype(float)
        target_chroma = target[entry_chroma_cols].values.astype(float)

        source_mfcc = source[mfcc_cols].values.astype(float)
        target_mfcc = target[mfcc_cols].values.astype(float)

        chroma_sim_raw = cos_sim(source_chroma, target_chroma)
        mfcc_sim = cos_sim(source_mfcc, target_mfcc)

        energy_change = float(target["entry_rms_mean"]) - float(source["exit_rms_mean"])
        energy_abs_diff = abs(energy_change)
        energy_score = 1.0 - min(energy_abs_diff / 0.15, 1.0)

        # v2 baseline: theory prior is primary; chroma/audio features refine it.
        baseline_score = (
            0.45 * prior
            + 0.25 * chroma_sim_raw
            + 0.10 * energy_score
            + 0.10 * mfcc_sim
            + 0.05 * (1.0 - min(float(source["exit_chroma_entropy"]) / 4.0, 1.0))
            + 0.05 * (1.0 - min(float(target["entry_chroma_entropy"]) / 4.0, 1.0))
        )

        row = {
            "transition_id": f"TC{transition_counter:07d}",
            "source_section_id": source["section_id"],
            "target_section_id": target["section_id"],
            "source_track_id": source["track_id"],
            "target_track_id": target["track_id"],
            "source_section_label": source["section_label"],
            "target_section_label": target["section_label"],

            "source_exit_key": source_key,
            "source_exit_mode": source_mode,
            "target_entry_key": target_key,
            "target_entry_mode": target_mode,

            **harmonic,

            "theory_prior_score": prior,
            "sampling_bucket": None,

            # Keep pitch fields for compatibility with preview code.
            # In this interval-prior version, previews are unshifted by default.
            "plan_type": "interval_prior",
            "shared_key": "",
            "shared_mode": "",
            "source_pitch_shift": 0,
            "target_pitch_shift": 0,
            "preserves_source_key": 1,
            "preserves_target_key": 1,
            "total_abs_pitch_shift": 0,

            "chroma_sim_raw": chroma_sim_raw,
            "chroma_sim_after_shift": chroma_sim_raw,
            "mfcc_sim": mfcc_sim,
            "energy_change": energy_change,
            "energy_abs_diff": energy_abs_diff,
            "source_exit_chroma_entropy": source["exit_chroma_entropy"],
            "target_entry_chroma_entropy": target["entry_chroma_entropy"],
            "baseline_score": baseline_score,
        }

        row["sampling_bucket"] = sampling_bucket(row)

        rows_by_source.setdefault(source["section_id"], []).append(row)
        transition_counter += 1

    # Per source, keep a mixture:
    #   mostly high-prior/high-score candidates
    #   plus epsilon exploration from lower-score / riskier intervals
    rng = np.random.default_rng(42)
    pruned = []

    for source_section_id, group_rows in rows_by_source.items():
        group = pd.DataFrame(group_rows)

        exploit_n = max(1, int(args.top_k_per_source * (1 - args.epsilon)))
        explore_n = args.top_k_per_source - exploit_n

        exploit = group.sort_values("baseline_score", ascending=False).head(exploit_n)

        remaining = group.drop(exploit.index, errors="ignore")

        if explore_n > 0 and len(remaining) > 0:
            # Encourage diversity across harmonic buckets/categories.
            explore_pool = remaining.copy()
            explore_pool["explore_weight"] = explore_pool["sampling_bucket"].map({
                "safe_theory": 0.5,
                "interesting_theory": 1.2,
                "risky_explore": 1.0,
            }).fillna(1.0)

            probs = explore_pool["explore_weight"].values.astype(float)
            probs = probs / probs.sum()

            n = min(explore_n, len(explore_pool))
            chosen_idx = rng.choice(explore_pool.index.values, size=n, replace=False, p=probs)
            explore = explore_pool.loc[chosen_idx].drop(columns=["explore_weight"])
        else:
            explore = pd.DataFrame(columns=group.columns)

        pruned_group = pd.concat([exploit, explore], ignore_index=True)
        pruned.extend(pruned_group.to_dict("records"))

    candidates = pd.DataFrame(pruned)

    if len(candidates) > args.max_candidates:
        candidates = candidates.sort_values("baseline_score", ascending=False).head(args.max_candidates)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_path, index=False)

    print(f"Saved {len(candidates)} transition candidates to {out_path}")

    print("\nHarmonic categories:")
    print(candidates["harmonic_category"].value_counts())

    print("\nSampling buckets:")
    print(candidates["sampling_bucket"].value_counts())

    print("\nTop candidates:")
    print(candidates.sort_values("baseline_score", ascending=False).head(15)[[
        "transition_id",
        "source_section_id",
        "target_section_id",
        "source_exit_key",
        "source_exit_mode",
        "target_entry_key",
        "target_entry_mode",
        "raw_interval_semitones",
        "harmonic_category",
        "theory_prior_score",
        "chroma_sim_raw",
        "baseline_score",
        "sampling_bucket",
    ]])


if __name__ == "__main__":
    main()