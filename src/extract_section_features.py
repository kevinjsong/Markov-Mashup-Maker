from pathlib import Path
import argparse

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


SR = 22050
N_MFCC = 20
ENTRY_SECONDS = 6.0
EXIT_SECONDS = 6.0


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


def chroma_entropy(chroma_mean):
    x = np.asarray(chroma_mean, dtype=float)
    total = np.sum(x)

    if total <= 0:
        return 0.0

    p = x / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


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


def safe_audio_load(path, start_time, end_time, sr=SR):
    duration = max(0.1, float(end_time) - float(start_time))

    y, sr = librosa.load(
        path,
        sr=sr,
        mono=True,
        offset=max(0.0, float(start_time)),
        duration=duration,
    )

    return y, sr


def extract_audio_features(y, sr):
    if len(y) == 0:
        raise ValueError("Empty audio segment")

    section_duration = librosa.get_duration(y=y, sr=sr)

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.asarray(tempo).squeeze())
    except Exception:
        tempo = np.nan

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    key, mode, key_confidence = estimate_key_from_chroma(chroma_mean)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    rms = librosa.feature.rms(y=y)
    rms_frames = rms.flatten()
    rms_mean = float(np.mean(rms_frames))
    rms_std = float(np.std(rms_frames))

    if len(rms_frames) > 1:
        xs = np.arange(len(rms_frames))
        energy_slope = float(np.polyfit(xs, rms_frames, deg=1)[0])
    else:
        energy_slope = 0.0

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = float(np.mean(centroid))
    spectral_centroid_std = float(np.std(centroid))

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = float(np.mean(zcr))

    features = {
        "duration": float(section_duration),
        "tempo": tempo,
        "key": key,
        "mode": mode,
        "key_confidence": key_confidence,
        "chroma_entropy": chroma_entropy(chroma_mean),
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "energy_slope": energy_slope,
        "spectral_centroid_mean": spectral_centroid_mean,
        "spectral_centroid_std": spectral_centroid_std,
        "zcr_mean": zcr_mean,
    }

    for i, value in enumerate(chroma_mean):
        features[f"chroma_{i}"] = float(value)

    for i, value in enumerate(mfcc_mean):
        features[f"mfcc_mean_{i}"] = float(value)

    for i, value in enumerate(mfcc_std):
        features[f"mfcc_std_{i}"] = float(value)

    return features


def prefix_features(features, prefix):
    return {f"{prefix}_{k}": v for k, v in features.items()}


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
        "--section_features_csv",
        default="data/training_processed/section_features.csv",
    )
    parser.add_argument(
        "--boundary_features_csv",
        default="data/training_processed/boundary_features.csv",
    )
    parser.add_argument(
        "--entry_seconds",
        type=float,
        default=ENTRY_SECONDS,
    )
    parser.add_argument(
        "--exit_seconds",
        type=float,
        default=EXIT_SECONDS,
    )

    args = parser.parse_args()

    tracks = pd.read_csv(args.tracks_csv)
    sections = pd.read_csv(args.sections_csv)

    tracks_by_id = tracks.set_index("track_id")

    section_rows = []
    boundary_rows = []

    for _, section in tqdm(sections.iterrows(), total=len(sections)):
        section_id = section["section_id"]
        track_id = section["track_id"]

        if track_id not in tracks_by_id.index:
            print(f"Skipping {section_id}: unknown track_id {track_id}")
            continue

        track = tracks_by_id.loc[track_id]
        path = track["path"]

        start_time = float(section["start_time"])
        end_time = float(section["end_time"])

        try:
            y_body, sr = safe_audio_load(path, start_time, end_time)
            body_features = extract_audio_features(y_body, sr)

            section_row = {
                "section_id": section_id,
                "track_id": track_id,
                "section_label": section["section_label"],
                "start_time": start_time,
                "end_time": end_time,
                "hook_score": section.get("hook_score", 1),
                **body_features,
            }
            section_rows.append(section_row)

            entry_start = start_time
            entry_end = min(end_time, start_time + args.entry_seconds)
            y_entry, sr = safe_audio_load(path, entry_start, entry_end)
            entry_features = extract_audio_features(y_entry, sr)

            exit_start = max(start_time, end_time - args.exit_seconds)
            exit_end = end_time
            y_exit, sr = safe_audio_load(path, exit_start, exit_end)
            exit_features = extract_audio_features(y_exit, sr)

            boundary_row = {
                "section_id": section_id,
                "track_id": track_id,
                "section_label": section["section_label"],
                "start_time": start_time,
                "end_time": end_time,
                "entry_start_time": entry_start,
                "entry_end_time": entry_end,
                "exit_start_time": exit_start,
                "exit_end_time": exit_end,
                **prefix_features(entry_features, "entry"),
                **prefix_features(exit_features, "exit"),
            }
            boundary_rows.append(boundary_row)

        except Exception as e:
            print(f"Skipping section {section_id}: {e}")

    section_features = pd.DataFrame(section_rows)
    boundary_features = pd.DataFrame(boundary_rows)

    section_out = Path(args.section_features_csv)
    boundary_out = Path(args.boundary_features_csv)

    section_out.parent.mkdir(parents=True, exist_ok=True)
    boundary_out.parent.mkdir(parents=True, exist_ok=True)

    section_features.to_csv(section_out, index=False)
    boundary_features.to_csv(boundary_out, index=False)

    print(f"Saved section features to {section_out}")
    print(f"Shape: {section_features.shape}")
    print(section_features.head())

    print(f"\nSaved boundary features to {boundary_out}")
    print(f"Shape: {boundary_features.shape}")
    print(boundary_features.head())


if __name__ == "__main__":
    main()