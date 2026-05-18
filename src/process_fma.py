from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


RAW_DIR = Path("data/raw")
AUDIO_DIR = RAW_DIR / "fma_small"
METADATA_PATH = RAW_DIR / "fma_metadata" / "tracks.csv"

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fma_track_path(track_id: int) -> Path:
    """
    FMA stores files like:
    fma_small/000/000002.mp3
    fma_small/012/012345.mp3
    """
    tid = f"{track_id:06d}"
    return AUDIO_DIR / tid[:3] / f"{tid}.mp3"


def load_tracks_metadata() -> pd.DataFrame:
    """
    FMA tracks.csv has a multi-row header.
    This loads it and flattens the columns we care about.
    """
    tracks = pd.read_csv(METADATA_PATH, index_col=0, header=[0, 1])

    rows = []

    for track_id, row in tracks.iterrows():
        path = fma_track_path(int(track_id))

        # Keep only files that actually exist in fma_small.
        if not path.exists():
            continue

        title = row.get(("track", "title"), None)
        artist = row.get(("artist", "name"), None)
        subset = row.get(("set", "subset"), None)
        split = row.get(("set", "split"), None)
        genre_top = row.get(("track", "genre_top"), None)

        rows.append(
            {
                "track_id": int(track_id),
                "title": title,
                "artist": artist,
                "subset": subset,
                "split": split,
                "genre_top": genre_top,
                "path": str(path),
            }
        )

    return pd.DataFrame(rows)


def get_audio_duration(path: str) -> float:
    try:
        return float(librosa.get_duration(path=path))
    except Exception:
        return np.nan


def make_full_clip_segment(track_id: int, duration: float):
    """
    Treat each FMA-small 30-second clip as one segment.
    """
    return [
        {
            "segment_id": f"{track_id}_full",
            "track_id": track_id,
            "start_time": 0.0,
            "end_time": duration,
            "segment_position": "full_clip",
        }
    ]


def extract_segment_features(
    path: str,
    start_time: float,
    end_time: float,
    sr: int = 22050,
):
    """
    Extract simple audio features from one segment.

    Output:
    - tempo: rhythm-ish
    - chroma: harmony/key-ish
    - MFCC: timbre/texture
    - RMS: energy/loudness
    - spectral centroid: brightness
    """
    duration = end_time - start_time

    y, sr = librosa.load(
        path,
        sr=sr,
        mono=True,
        offset=start_time,
        duration=duration,
    )

    if len(y) == 0:
        raise ValueError("Empty audio segment")

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).squeeze())

    # Harmony-ish features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Timbre/texture features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Energy
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    # Brightness
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = float(np.mean(centroid))
    spectral_centroid_std = float(np.std(centroid))

    features = {
        "tempo": tempo,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "spectral_centroid_mean": spectral_centroid_mean,
        "spectral_centroid_std": spectral_centroid_std,
    }

    for i, value in enumerate(chroma_mean):
        features[f"chroma_{i}"] = float(value)

    for i, value in enumerate(mfcc_mean):
        features[f"mfcc_mean_{i}"] = float(value)

    for i, value in enumerate(mfcc_std):
        features[f"mfcc_std_{i}"] = float(value)

    return features


def main(max_tracks: int = 100):
    print("Loading FMA metadata...")
    tracks_df = load_tracks_metadata()

    print(f"Found {len(tracks_df)} audio files in fma_small.")

    target_genres = ["Pop", "Folk"]
    tracks_df = tracks_df[tracks_df["genre_top"].isin(target_genres)].copy()

    print(f"Keeping {len(tracks_df)} tracks after genre filter: {target_genres}")

    # Start small. You can raise this later.
    tracks_df = tracks_df.head(max_tracks).copy()

    print("Computing durations...")
    tracks_df["duration"] = [
        get_audio_duration(path) for path in tqdm(tracks_df["path"])
    ]

    # Skip broken or very short tracks.
    tracks_df = tracks_df.dropna(subset=["duration"])
    tracks_df = tracks_df[tracks_df["duration"] >= 15].copy()

    print(f"Keeping {len(tracks_df)} usable tracks.")

    tracks_out = PROCESSED_DIR / "tracks.csv"
    tracks_df.to_csv(tracks_out, index=False)

    print("Splitting tracks into thirds...")
    segment_rows = []

    for _, row in tracks_df.iterrows():
        segment_rows.extend(
            make_full_clip_segment(
                track_id=int(row["track_id"]),
                duration=float(row["duration"]),
            )
        )

    segments_df = pd.DataFrame(segment_rows)
    segments_out = PROCESSED_DIR / "segments.csv"
    segments_df.to_csv(segments_out, index=False)

    print(f"Created {len(segments_df)} segments.")

    print("Extracting segment features...")
    path_lookup = dict(zip(tracks_df["track_id"], tracks_df["path"]))

    feature_rows = []

    for _, seg in tqdm(segments_df.iterrows(), total=len(segments_df)):
        track_id = int(seg["track_id"])
        path = path_lookup[track_id]

        try:
            feats = extract_segment_features(
                path=path,
                start_time=float(seg["start_time"]),
                end_time=float(seg["end_time"]),
            )

            feature_rows.append(
                {
                    "segment_id": seg["segment_id"],
                    "track_id": track_id,
                    "segment_position": seg["segment_position"],
                    **feats,
                }
            )

        except Exception as e:
            print(f"Skipping segment {seg['segment_id']} due to error: {e}")

    features_df = pd.DataFrame(feature_rows)
    features_out = PROCESSED_DIR / "segment_features.csv"
    features_df.to_csv(features_out, index=False)

    print("\nDone.")
    print(f"Saved: {tracks_out}")
    print(f"Saved: {segments_out}")
    print(f"Saved: {features_out}")
    print(f"tracks shape: {tracks_df.shape}")
    print(f"segments shape: {segments_df.shape}")
    print(f"features shape: {features_df.shape}")


if __name__ == "__main__":
    main(max_tracks=100)