from pathlib import Path
from datetime import datetime
import subprocess
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


PROCESSED_DIR = Path("data/processed")
PREVIEW_DIR = Path("outputs/previews")
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

SR = 22050
SOURCE_TAIL_SECONDS = 5.0
TARGET_HEAD_SECONDS = 5.0
CROSSFADE_SECONDS = 0.75


def load_audio_slice(path, start_time, duration, sr=SR):
    y, sr = librosa.load(
        path,
        sr=sr,
        mono=True,
        offset=max(0.0, start_time),
        duration=max(0.1, duration),
    )
    return y


def normalize_audio(y):
    if len(y) == 0:
        return y

    max_abs = np.max(np.abs(y))

    if max_abs == 0:
        return y

    return y / max_abs * 0.9


def crossfade(source_tail, target_head, sr=SR, crossfade_seconds=CROSSFADE_SECONDS):
    fade_len = int(crossfade_seconds * sr)

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


def create_transition_preview(
    transition_id,
    comparison_id,
    option_label,
    transitions,
    segments,
    tracks,
):
    """
    Create a short audio preview for one transition:
    last 5 seconds of source segment + crossfade + first 5 seconds of target segment.
    """
    out_path = PREVIEW_DIR / f"{comparison_id}_{option_label}_{transition_id}.wav"

    # Reuse if already created.
    if out_path.exists():
        return out_path

    transition = transitions.loc[transition_id]

    source_segment = segments.loc[transition["source_segment_id"]]
    target_segment = segments.loc[transition["target_segment_id"]]

    source_track = tracks.loc[int(source_segment["track_id"])]
    target_track = tracks.loc[int(target_segment["track_id"])]

    source_path = source_track["path"]
    target_path = target_track["path"]

    source_start = float(source_segment["start_time"])
    source_end = float(source_segment["end_time"])

    target_start = float(target_segment["start_time"])
    target_end = float(target_segment["end_time"])

    # Last N seconds of source segment.
    source_tail_start = max(source_start, source_end - SOURCE_TAIL_SECONDS)
    source_tail_duration = source_end - source_tail_start

    # First N seconds of target segment.
    target_head_start = target_start
    target_head_duration = min(TARGET_HEAD_SECONDS, target_end - target_start)

    source_tail = load_audio_slice(
        source_path,
        source_tail_start,
        source_tail_duration,
    )

    target_head = load_audio_slice(
        target_path,
        target_head_start,
        target_head_duration,
    )

    preview = crossfade(source_tail, target_head)
    preview = normalize_audio(preview)

    sf.write(out_path, preview, SR)

    return out_path


def play_audio(path):
    """
    macOS audio playback from terminal.
    """
    subprocess.run(["afplay", str(path)])


def describe_transition(transition_id, transitions, segments, tracks):
    transition = transitions.loc[transition_id]

    source_segment_id = transition["source_segment_id"]
    target_segment_id = transition["target_segment_id"]

    source_segment = segments.loc[source_segment_id]
    target_segment = segments.loc[target_segment_id]

    source_track = tracks.loc[int(source_segment["track_id"])]
    target_track = tracks.loc[int(target_segment["track_id"])]

    source_title = source_track.get("title", "Unknown")
    source_artist = source_track.get("artist", "Unknown")
    target_title = target_track.get("title", "Unknown")
    target_artist = target_track.get("artist", "Unknown")

    return f"""
{transition_id}
  Source:
    {source_title} — {source_artist}
    segment: {source_segment["segment_position"]}
    time: {float(source_segment["start_time"]):.1f}s–{float(source_segment["end_time"]):.1f}s

  Target:
    {target_title} — {target_artist}
    segment: {target_segment["segment_position"]}
    time: {float(target_segment["start_time"]):.1f}s–{float(target_segment["end_time"]):.1f}s

  Features:
    baseline_score: {float(transition["baseline_score"]):.3f}
    tempo_diff:     {float(transition["tempo_diff"]):.3f}
    chroma_sim:     {float(transition["chroma_sim"]):.3f}
    mfcc_sim:       {float(transition["mfcc_sim"]):.3f}
"""


def main():
    transitions_path = PROCESSED_DIR / "candidate_transitions.csv"
    segments_path = PROCESSED_DIR / "segments.csv"
    tracks_path = PROCESSED_DIR / "tracks.csv"
    queue_path = PROCESSED_DIR / "feedback_queue.csv"
    preferences_path = PROCESSED_DIR / "preferences.csv"

    for path in [transitions_path, segments_path, tracks_path, queue_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    transitions = pd.read_csv(transitions_path).set_index("transition_id")
    segments = pd.read_csv(segments_path).set_index("segment_id")
    tracks = pd.read_csv(tracks_path).set_index("track_id")
    queue = pd.read_csv(queue_path)

    if preferences_path.exists():
        preferences = pd.read_csv(preferences_path)
        labeled_comparisons = set(preferences["comparison_id"])
    else:
        preferences = pd.DataFrame(
            columns=[
                "preference_id",
                "comparison_id",
                "source_segment_id",
                "transition_a",
                "transition_b",
                "winner",
                "created_at",
            ]
        )
        labeled_comparisons = set()

    new_rows = []

    for _, row in queue.iterrows():
        comparison_id = row["comparison_id"]

        if comparison_id in labeled_comparisons:
            continue

        transition_a = row["transition_a"]
        transition_b = row["transition_b"]

        print("\n" + "=" * 80)
        print(f"Comparison: {comparison_id}")
        print("=" * 80)

        print("\nOption A:")
        print(describe_transition(transition_a, transitions, segments, tracks))

        print("\nOption B:")
        print(describe_transition(transition_b, transitions, segments, tracks))

        print("\nCreating/listening to previews...")

        preview_a = create_transition_preview(
            transition_id=transition_a,
            comparison_id=comparison_id,
            option_label="A",
            transitions=transitions,
            segments=segments,
            tracks=tracks,
        )

        preview_b = create_transition_preview(
            transition_id=transition_b,
            comparison_id=comparison_id,
            option_label="B",
            transitions=transitions,
            segments=segments,
            tracks=tracks,
        )

        while True:
            print("\nPlaying Option A...")
            play_audio(preview_a)

            time.sleep(0.5)

            print("Playing Option B...")
            play_audio(preview_b)

            choice = input(
                "\nWhich transition is better? "
                "Enter A, B, R to replay, S to skip, or Q to quit: "
            ).strip().lower()

            if choice in {"a", "b", "r", "s", "q"}:
                break

            print("Invalid input. Please enter A, B, R, S, or Q.")

        if choice == "q":
            break

        if choice == "r":
            # Replay the same comparison.
            while choice == "r":
                print("\nReplaying Option A...")
                play_audio(preview_a)

                time.sleep(0.5)

                print("Replaying Option B...")
                play_audio(preview_b)

                choice = input(
                    "\nWhich transition is better? "
                    "Enter A, B, R to replay, S to skip, or Q to quit: "
                ).strip().lower()

            if choice == "q":
                break

        if choice == "s":
            continue

        if choice == "a":
            winner = transition_a
        elif choice == "b":
            winner = transition_b
        else:
            print("Invalid input after replay. Skipping.")
            continue

        preference_id = f"P{len(preferences) + len(new_rows):06d}"

        new_rows.append(
            {
                "preference_id": preference_id,
                "comparison_id": comparison_id,
                "source_segment_id": row["source_segment_id"],
                "transition_a": transition_a,
                "transition_b": transition_b,
                "winner": winner,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

        all_preferences = pd.concat(
            [preferences, pd.DataFrame(new_rows)],
            ignore_index=True,
        )

        all_preferences.to_csv(preferences_path, index=False)

        print(f"Saved preference: {preference_id}")

    print(f"\nDone. Preferences saved to {preferences_path}")


if __name__ == "__main__":
    main()