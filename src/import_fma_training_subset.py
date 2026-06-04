from pathlib import Path
import argparse

import librosa
import pandas as pd
from tqdm import tqdm


RAW_DIR = Path("data/raw")
DEFAULT_AUDIO_DIR = RAW_DIR / "fma_small"
DEFAULT_METADATA_CSV = RAW_DIR / "fma_metadata" / "tracks.csv"
DEFAULT_OUTPUT_DIR = Path("data/training_processed")


def fma_track_path(audio_dir: Path, track_id: int) -> Path:
    """
    FMA stores files like:
      fma_small/000/000010.mp3
      fma_small/012/012345.mp3
    """
    tid = f"{track_id:06d}"
    return audio_dir / tid[:3] / f"{tid}.mp3"


def get_duration(path: Path) -> float:
    try:
        return float(librosa.get_duration(path=str(path)))
    except Exception as e:
        print(f"Could not read duration for {path}: {e}")
        return float("nan")


def load_fma_tracks(metadata_csv: Path, audio_dir: Path) -> pd.DataFrame:
    fma = pd.read_csv(metadata_csv, index_col=0, header=[0, 1])

    rows = []

    for track_id, row in fma.iterrows():
        track_id = int(track_id)
        path = fma_track_path(audio_dir, track_id)

        if not path.exists():
            continue

        rows.append(
            {
                "track_id": f"FMA_{track_id:06d}",
                "source_track_id": track_id,
                "title": row.get(("track", "title"), ""),
                "artist": row.get(("artist", "name"), ""),
                "genre_top": row.get(("track", "genre_top"), ""),
                "subset": row.get(("set", "subset"), ""),
                "split": row.get(("set", "split"), ""),
                "path": str(path),
                "source_dataset": "fma",
                "tags_raw": "",
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--audio_dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--genres", nargs="+", default=["Pop", "Folk"])
    parser.add_argument("--max_tracks", type=int, default=300)

    args = parser.parse_args()

    metadata_csv = Path(args.metadata_csv)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing FMA metadata CSV: {metadata_csv}")

    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing FMA audio directory: {audio_dir}")

    print("Loading FMA metadata...")
    tracks = load_fma_tracks(metadata_csv, audio_dir)

    print(f"Found {len(tracks)} available FMA audio files.")

    if args.genres:
        tracks = tracks[tracks["genre_top"].isin(args.genres)].copy()
        print(f"Filtered to genres {args.genres}: {len(tracks)} tracks")

    tracks = tracks.head(args.max_tracks).copy()

    print("Computing durations...")
    durations = []
    for path in tqdm(tracks["path"]):
        durations.append(get_duration(Path(path)))

    tracks["duration"] = durations
    tracks = tracks.dropna(subset=["duration"])
    tracks = tracks[tracks["duration"] > 5.0].copy()

    out_path = output_dir / "tracks.csv"
    tracks.to_csv(out_path, index=False)

    print(f"Saved {len(tracks)} tracks to {out_path}")
    print(tracks[["track_id", "title", "artist", "genre_top", "path", "duration"]].head())
    print("\nGenre counts:")
    print(tracks["genre_top"].value_counts())


if __name__ == "__main__":
    main()