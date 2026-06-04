from pathlib import Path
import argparse

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


SR = 22050

KEY_NAMES = [
    "C", "C#/Db", "D", "D#/Eb", "E", "F",
    "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"
]

MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88
])

MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17
])


def normalize_vector(x):
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


def estimate_key_from_chroma(chroma_mean):
    chroma_norm = normalize_vector(chroma_mean)

    best_score = -np.inf
    best_key = None
    best_mode = None

    for i in range(12):
        major_profile = normalize_vector(np.roll(MAJOR_PROFILE, i))
        minor_profile = normalize_vector(np.roll(MINOR_PROFILE, i))

        major_score = float(np.dot(chroma_norm, major_profile))
        minor_score = float(np.dot(chroma_norm, minor_profile))

        if major_score > best_score:
            best_score = major_score
            best_key = KEY_NAMES[i]
            best_mode = "major"

        if minor_score > best_score:
            best_score = minor_score
            best_key = KEY_NAMES[i]
            best_mode = "minor"

    return best_key, best_mode, best_score


def estimate_window_key(y, sr):
    if len(y) == 0:
        return None, None, 0.0

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    return estimate_key_from_chroma(chroma_mean)


def mode_key_label(key, mode):
    return f"{key}_{mode}"


def smooth_labels(labels, min_persist_windows=2):
    """
    Remove isolated one-window flips.

    Example:
      A A C A A -> A A A A A if min_persist_windows=2
    """
    if not labels:
        return labels

    smoothed = labels[:]
    n = len(labels)

    i = 0
    while i < n:
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1

        run_len = j - i

        if run_len < min_persist_windows:
            prev_label = smoothed[i - 1] if i > 0 else None
            next_label = labels[j] if j < n else None

            replacement = None
            if prev_label is not None and next_label is not None and prev_label == next_label:
                replacement = prev_label
            elif prev_label is not None:
                replacement = prev_label
            elif next_label is not None:
                replacement = next_label

            if replacement is not None:
                for k in range(i, j):
                    smoothed[k] = replacement

        i = j

    return smoothed


def merge_short_sections(sections, min_section_sec):
    """
    Merge tiny sections into neighboring sections.
    """
    if len(sections) <= 1:
        return sections

    merged = []

    for sec in sections:
        duration = sec["end_time"] - sec["start_time"]

        if duration >= min_section_sec or not merged:
            merged.append(sec)
        else:
            # Merge short section into previous.
            merged[-1]["end_time"] = sec["end_time"]
            merged[-1]["confidence"] = min(merged[-1]["confidence"], sec["confidence"])
            merged[-1]["section_label"] = (
                merged[-1]["section_label"]
                if merged[-1]["section_label"] == sec["section_label"]
                else merged[-1]["section_label"] + "_merged"
            )

    # If final section is short, merge backward.
    if len(merged) > 1:
        final = merged[-1]
        if final["end_time"] - final["start_time"] < min_section_sec:
            prev = merged[-2]
            prev["end_time"] = final["end_time"]
            prev["confidence"] = min(prev["confidence"], final["confidence"])
            prev["section_label"] = (
                prev["section_label"]
                if prev["section_label"] == final["section_label"]
                else prev["section_label"] + "_merged"
            )
            merged = merged[:-1]

    return merged


def make_key_change_sections_for_track(
    track,
    window_sec,
    hop_sec,
    min_section_sec,
    min_persist_windows,
    confidence_margin,
):
    track_id = track["track_id"]
    path = track["path"]
    duration = float(track["duration"])

    windows = []
    t = 0.0

    while t < duration:
        end = min(t + window_sec, duration)

        if end - t < max(4.0, window_sec * 0.5):
            break

        y, sr = librosa.load(
            path,
            sr=SR,
            mono=True,
            offset=t,
            duration=end - t,
        )

        key, mode, conf = estimate_window_key(y, sr)

        windows.append(
            {
                "start": t,
                "end": end,
                "key": key,
                "mode": mode,
                "confidence": conf,
                "label": mode_key_label(key, mode),
            }
        )

        t += hop_sec

    if not windows:
        return [
            {
                "section_id": f"{track_id}_ksec001",
                "track_id": track_id,
                "section_label": "key_region_unknown",
                "start_time": 0.0,
                "end_time": duration,
                "hook_score": 1,
                "source": "key_change",
                "confidence": 0.0,
            }
        ]

    labels = [w["label"] for w in windows]
    labels = smooth_labels(labels, min_persist_windows=min_persist_windows)

    # Group consecutive windows by smoothed key/mode label.
    rough_sections = []

    start_idx = 0

    for i in range(1, len(windows)):
        if labels[i] != labels[start_idx]:
            group = windows[start_idx:i]
            key, mode = labels[start_idx].rsplit("_", 1)

            rough_sections.append(
                {
                    "key": key,
                    "mode": mode,
                    "start_time": group[0]["start"],
                    "end_time": group[-1]["end"],
                    "confidence": float(np.mean([g["confidence"] for g in group])),
                }
            )

            start_idx = i

    group = windows[start_idx:]
    key, mode = labels[start_idx].rsplit("_", 1)

    rough_sections.append(
        {
            "key": key,
            "mode": mode,
            "start_time": group[0]["start"],
            "end_time": min(group[-1]["end"], duration),
            "confidence": float(np.mean([g["confidence"] for g in group])),
        }
    )

    # If confidence is low overall or no meaningful change, this naturally becomes one region.
    cleaned = []

    for sec in rough_sections:
        cleaned.append(
            {
                "section_id": "",  # filled after merging
                "track_id": track_id,
                "section_label": f"key_region_{sec['key']}_{sec['mode']}",
                "start_time": round(sec["start_time"], 3),
                "end_time": round(sec["end_time"], 3),
                "hook_score": 1,
                "source": "key_change",
                "confidence": round(sec["confidence"], 4),
            }
        )

    cleaned = merge_short_sections(cleaned, min_section_sec=min_section_sec)

    # If all windows are unstable/low confidence, still keep usable sections;
    # don't over-delete. But mark low confidence.
    final = []
    for idx, sec in enumerate(cleaned, start=1):
        sec["section_id"] = f"{track_id}_ksec{idx:03d}"
        final.append(sec)

    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracks_csv",
        default="data/user_uploads/processed/tracks.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="data/user_uploads/processed/sections.csv",
    )
    parser.add_argument("--window_sec", type=float, default=12.0)
    parser.add_argument("--hop_sec", type=float, default=6.0)
    parser.add_argument("--min_section_sec", type=float, default=20.0)
    parser.add_argument("--min_persist_windows", type=int, default=2)
    parser.add_argument("--confidence_margin", type=float, default=0.0)

    args = parser.parse_args()

    tracks = pd.read_csv(args.tracks_csv)

    all_sections = []

    for _, track in tqdm(tracks.iterrows(), total=len(tracks)):
        sections = make_key_change_sections_for_track(
            track=track,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            min_section_sec=args.min_section_sec,
            min_persist_windows=args.min_persist_windows,
            confidence_margin=args.confidence_margin,
        )
        all_sections.extend(sections)

    sections_df = pd.DataFrame(all_sections)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sections_df.to_csv(out_path, index=False)

    print(f"Saved {len(sections_df)} key-change sections to {out_path}")
    print(sections_df.head(20))

    print("\nSections per track:")
    print(sections_df.groupby("track_id").size().describe())


if __name__ == "__main__":
    main()