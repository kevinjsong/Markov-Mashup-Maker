from pathlib import Path
from datetime import datetime
import argparse
import subprocess
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


SR = 22050
SOURCE_EXIT_SECONDS = 6.0
TARGET_ENTRY_SECONDS = 6.0
CROSSFADE_SECONDS = 1.0


def normalize_audio(y):
    if len(y) == 0:
        return y

    max_abs = np.max(np.abs(y))
    if max_abs == 0:
        return y

    return y / max_abs * 0.9


def crossfade(source_tail, target_head, sr=SR):
    fade_len = int(CROSSFADE_SECONDS * sr)

    if len(source_tail) < fade_len or len(target_head) < fade_len:
        return np.concatenate([source_tail, target_head])

    source_main = source_tail[:-fade_len]
    source_fade = source_tail[-fade_len:]

    target_fade = target_head[:fade_len]
    target_main = target_head[fade_len:]

    fade_out = np.linspace(1.0, 0.0, fade_len)
    fade_in = np.linspace(0.0, 1.0, fade_len)

    blended = source_fade * fade_out + target_fade * fade_in

    return np.concatenate([source_main, blended, target_main])


def apply_pitch_shift(y, semitone_shift, sr=SR):
    if pd.isna(semitone_shift) or str(semitone_shift).strip() == "":
        semitone_shift = 0

    semitone_shift = int(float(semitone_shift))

    if semitone_shift == 0:
        return y

    return librosa.effects.pitch_shift(
        y,
        sr=sr,
        n_steps=semitone_shift,
    )


def load_audio_window(path, start_time, end_time):
    duration = max(0.1, float(end_time) - float(start_time))

    y, sr = librosa.load(
        path,
        sr=SR,
        mono=True,
        offset=max(0.0, float(start_time)),
        duration=duration,
    )

    return y


def create_transition_preview(candidate, tracks_by_id, sections_by_id, preview_dir):
    """
    Create preview:

      last 6 sec of source section, pitch shifted
      + crossfade
      first 6 sec of target section, pitch shifted

    The pitch shifts come from the transition candidate's harmonic plan.
    """
    transition_id = candidate["transition_id"]
    out_path = preview_dir / f"{transition_id}.wav"

    if out_path.exists():
        return out_path

    source_section_id = candidate["source_section_id"]
    target_section_id = candidate["target_section_id"]

    if source_section_id not in sections_by_id.index:
        raise KeyError(f"source_section_id not found in sections.csv: {source_section_id}")

    if target_section_id not in sections_by_id.index:
        raise KeyError(f"target_section_id not found in sections.csv: {target_section_id}")

    source_section = sections_by_id.loc[source_section_id]
    target_section = sections_by_id.loc[target_section_id]

    source_track = tracks_by_id.loc[source_section["track_id"]]
    target_track = tracks_by_id.loc[target_section["track_id"]]

    source_start = float(source_section["start_time"])
    source_end = float(source_section["end_time"])

    target_start = float(target_section["start_time"])
    target_end = float(target_section["end_time"])

    source_exit_start = max(source_start, source_end - SOURCE_EXIT_SECONDS)
    source_exit_end = source_end

    target_entry_start = target_start
    target_entry_end = min(target_end, target_start + TARGET_ENTRY_SECONDS)

    source_audio = load_audio_window(
        source_track["path"],
        source_exit_start,
        source_exit_end,
    )

    target_audio = load_audio_window(
        target_track["path"],
        target_entry_start,
        target_entry_end,
    )

    source_audio = apply_pitch_shift(
        source_audio,
        candidate.get("source_pitch_shift", 0),
    )

    target_audio = apply_pitch_shift(
        target_audio,
        candidate.get("target_pitch_shift", 0),
    )

    preview = crossfade(source_audio, target_audio)
    preview = normalize_audio(preview)

    preview_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, preview, SR)

    return out_path


def play_audio(path):
    try:
        subprocess.run(["afplay", str(path)], check=False)
    except FileNotFoundError:
        print(f"afplay not found. Open this preview manually: {path}")


def describe_candidate(c):
    """
    Human-readable description of a transition candidate.

    Supports the newer interval-prior candidate schema.
    """
    return f"""
{c["transition_id"]}
  {c["source_section_id"]} → {c["target_section_id"]}

  source exit: {c.get("source_exit_key", "?")} {c.get("source_exit_mode", "?")}
  target entry: {c.get("target_entry_key", "?")} {c.get("target_entry_mode", "?")}

  interval: {c.get("interval_name", "?")}
  raw semitone interval: {c.get("raw_interval_semitones", "?")}
  signed semitone interval: {c.get("signed_interval_semitones", "?")}
  circle-of-fifths distance: {c.get("circle_of_fifths_distance", "?")}

  harmonic category: {c.get("harmonic_category", "?")}
  sampling bucket: {c.get("sampling_bucket", "?")}
  theory prior score: {float(c.get("theory_prior_score", 0.0)):.3f}

  chroma similarity: {float(c.get("chroma_sim_raw", c.get("chroma_sim_after_shift", 0.0))):.3f}
  energy change: {float(c.get("energy_change", 0.0)):+.4f}
  baseline score: {float(c.get("baseline_score", 0.0)):.3f}
"""


def make_feedback_pairs(candidates, max_pairs=50, seed=42, epsilon=0.5):
    """
    Safe/exploration feedback sampler.

    epsilon = probability of exploration.
    With epsilon=0.5:
      50% safe/theory-supported comparisons
      50% exploratory/random-diverse comparisons
    """
    rng = np.random.default_rng(seed)

    candidates = candidates.copy()

    if "harmonic_category" not in candidates.columns:
        candidates["harmonic_category"] = "unknown"

    if "baseline_score" not in candidates.columns:
        candidates["baseline_score"] = 0.0

    safe_categories = {
        "same_key",
        "circle_neighbor",
        "relative_major_to_minor",
        "relative_minor_to_major",
    }

    exploratory_categories = {
        "parallel_mode",
        "near_circle",
        "chromatic_mediant",
        "whole_step_shift",
        "half_step_shift",
        "tritone",
        "distant_other",
    }

    rows = []
    pair_counter = 0
    seen_pairs = set()

    sources = list(candidates["source_section_id"].unique())
    rng.shuffle(sources)

    for source_section_id in sources:
        group = candidates[candidates["source_section_id"] == source_section_id].copy()

        if len(group) < 2:
            continue

        safe_pool = group[group["harmonic_category"].isin(safe_categories)].copy()
        explore_pool = group[group["harmonic_category"].isin(exploratory_categories)].copy()

        # Fall back gracefully if a source lacks either pool.
        if len(safe_pool) < 2:
            safe_pool = group.copy()

        if len(explore_pool) < 2:
            explore_pool = group.copy()

        # epsilon = probability of exploration.
        do_safe = rng.random() >= epsilon

        if do_safe:
            safe_pool = safe_pool.sort_values("baseline_score", ascending=False)
            top_safe = safe_pool.head(min(16, len(safe_pool)))

            sampled = top_safe.sample(
                n=2,
                random_state=int(rng.integers(0, 1_000_000)),
            )

            comparison_type = "safe_key_50"

        else:
            categories = explore_pool["harmonic_category"].dropna().unique().tolist()

            if len(categories) >= 2:
                cat_a, cat_b = rng.choice(categories, size=2, replace=False)

                pool_a = explore_pool[explore_pool["harmonic_category"] == cat_a]
                pool_b = explore_pool[explore_pool["harmonic_category"] == cat_b]

                a = pool_a.sample(
                    n=1,
                    random_state=int(rng.integers(0, 1_000_000)),
                ).iloc[0]

                b = pool_b.sample(
                    n=1,
                    random_state=int(rng.integers(0, 1_000_000)),
                ).iloc[0]

                sampled = pd.DataFrame([a, b])

            else:
                sampled = explore_pool.sample(
                    n=2,
                    random_state=int(rng.integers(0, 1_000_000)),
                )

            comparison_type = "random_exploration_50"

        a = sampled.iloc[0]
        b = sampled.iloc[1]

        if a["transition_id"] == b["transition_id"]:
            continue

        pair_key = tuple(sorted([str(a["transition_id"]), str(b["transition_id"])]))

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)

        rows.append(
            {
                "comparison_id": f"CMP{pair_counter:06d}",
                "source_section_id": source_section_id,
                "transition_a": a["transition_id"],
                "transition_b": b["transition_id"],
                "comparison_type": comparison_type,
                "harmonic_category_a": a.get("harmonic_category", "unknown"),
                "harmonic_category_b": b.get("harmonic_category", "unknown"),
            }
        )

        pair_counter += 1

        if len(rows) >= max_pairs:
            break

    return pd.DataFrame(rows)


def get_next_preference_number(preferences):
    if preferences.empty or "preference_id" not in preferences.columns:
        return 0

    nums = []

    for value in preferences["preference_id"].dropna().astype(str):
        if value.startswith("PREF"):
            suffix = value.replace("PREF", "")
            if suffix.isdigit():
                nums.append(int(suffix))

    if not nums:
        return len(preferences)

    return max(nums) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracks_csv",
        default="data/training_processed/tracks.csv",
    )
    parser.add_argument(
        "--sections_csv",
        default="data/training_processed/sections.csv",
    )
    parser.add_argument(
        "--candidates_csv",
        default="data/training_processed/transition_candidates.csv",
    )
    parser.add_argument(
        "--preferences_csv",
        default="data/training_processed/preferences.csv",
    )
    parser.add_argument(
        "--preview_dir",
        default="outputs/previews",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--context",
        default="general",
        help="Legacy compatibility field. The current trainer ignores context.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.5,
        help="Probability of exploratory sampling. 0.5 gives 50/50 safe/exploratory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    tracks = pd.read_csv(args.tracks_csv)
    sections = pd.read_csv(args.sections_csv)
    candidates = pd.read_csv(args.candidates_csv)

    if candidates.empty:
        raise ValueError("No transition candidates found.")

    tracks_by_id = tracks.set_index("track_id")
    sections_by_id = sections.set_index("section_id")
    candidates_by_id = candidates.set_index("transition_id", drop=False)

    preferences_path = Path(args.preferences_csv)
    preview_dir = Path(args.preview_dir)

    if preferences_path.exists():
        preferences = pd.read_csv(preferences_path)

        # Keep old files compatible.
        if "context" not in preferences.columns:
            preferences["context"] = args.context

        labeled = set(preferences["comparison_id"].astype(str))
    else:
        preferences = pd.DataFrame(
            columns=[
                "preference_id",
                "comparison_id",
                "context",
                "transition_a",
                "transition_b",
                "winner",
                "label_type",
                "created_at",
            ]
        )
        labeled = set()

    feedback_pairs = make_feedback_pairs(
        candidates,
        max_pairs=args.max_pairs,
        seed=args.seed,
        epsilon=args.epsilon,
    )

    new_rows = []
    next_pref_num = get_next_preference_number(preferences)

    print("\nContext for this labeling session:")
    print(f"  {args.context}")
    print("\nSampling:")
    print(f"  epsilon={args.epsilon:.2f}")
    print(f"  safe probability={1.0 - args.epsilon:.2f}")
    print(f"  exploratory probability={args.epsilon:.2f}")

    print("\nChoices:")
    print("  A = option A better")
    print("  B = option B better")
    print("  N = neither / both bad")
    print("  G = both good")
    print("  R = replay")
    print("  S = skip")
    print("  Q = quit")

    for _, pair in feedback_pairs.iterrows():
        comparison_id = str(pair["comparison_id"])

        if comparison_id in labeled:
            continue

        transition_a = pair["transition_a"]
        transition_b = pair["transition_b"]

        cand_a = candidates_by_id.loc[transition_a]
        cand_b = candidates_by_id.loc[transition_b]

        print("\n" + "=" * 80)
        print(f"Comparison {comparison_id}")
        print(f"Context: {args.context}")
        print(f"Sampling type: {pair.get('comparison_type', 'unknown')}")
        print(f"A category: {pair.get('harmonic_category_a', 'unknown')}")
        print(f"B category: {pair.get('harmonic_category_b', 'unknown')}")
        print("=" * 80)

        print("\nOption A:")
        print(describe_candidate(cand_a))

        print("\nOption B:")
        print(describe_candidate(cand_b))

        print("\nCreating previews...")
        preview_a = create_transition_preview(
            cand_a,
            tracks_by_id,
            sections_by_id,
            preview_dir,
        )
        preview_b = create_transition_preview(
            cand_b,
            tracks_by_id,
            sections_by_id,
            preview_dir,
        )

        while True:
            print("\nPlaying A...")
            play_audio(preview_a)
            time.sleep(0.5)

            print("Playing B...")
            play_audio(preview_b)

            choice = input(
                "\nWhich transition is better? A/B/N neither/G both good/R replay/S skip/Q quit: "
            ).strip().lower()

            if choice in {"a", "b", "n", "g", "r", "s", "q"}:
                break

            print("Invalid input.")

        while choice == "r":
            print("\nReplaying A...")
            play_audio(preview_a)
            time.sleep(0.5)

            print("Replaying B...")
            play_audio(preview_b)

            choice = input(
                "\nWhich transition is better? A/B/N neither/G both good/R replay/S skip/Q quit: "
            ).strip().lower()

            if choice not in {"a", "b", "n", "g", "r", "s", "q"}:
                print("Invalid input.")
                choice = "r"

        if choice == "q":
            break

        if choice == "s":
            continue

        if choice == "a":
            winner = transition_a
            label_type = "pairwise_preference"
        elif choice == "b":
            winner = transition_b
            label_type = "pairwise_preference"
        elif choice == "n":
            winner = "neither"
            label_type = "reject_both"
        elif choice == "g":
            winner = "both_good"
            label_type = "accept_both"
        else:
            continue

        preference_id = f"PREF{next_pref_num:06d}"
        next_pref_num += 1

        new_rows.append(
            {
                "preference_id": preference_id,
                "comparison_id": comparison_id,
                "context": args.context,
                "transition_a": transition_a,
                "transition_b": transition_b,
                "winner": winner,
                "label_type": label_type,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

        all_preferences = pd.concat(
            [preferences, pd.DataFrame(new_rows)],
            ignore_index=True,
        )
        all_preferences.to_csv(preferences_path, index=False)

        labeled.add(comparison_id)

        print(f"Saved preference {preference_id}")

    all_preferences = pd.concat(
        [preferences, pd.DataFrame(new_rows)],
        ignore_index=True,
    )
    all_preferences.to_csv(preferences_path, index=False)

    print(f"\nDone. Preferences saved to {preferences_path}")
    print("\nLabel counts:")
    print(all_preferences["label_type"].value_counts())


if __name__ == "__main__":
    main()