from pathlib import Path
import argparse

import librosa
import pandas as pd


SUPPORTED_EXTS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aiff",
    ".aif",
}


def get_audio_duration(path: Path) -> float:
    """
    Return audio duration in seconds.

    librosa.get_duration(path=...) is fast when supported. If it fails,
    fall back to loading the file.
    """
    try:
        return float(librosa.get_duration(path=str(path)))
    except Exception:
        y, sr = librosa.load(str(path), sr=22050, mono=True)
        return float(len(y) / sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dir",
        required=True,
        help="Directory containing local audio files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where tracks.csv should be written.",
    )
    parser.add_argument(
        "--prefix",
        default="USER",
        help="Prefix for generated track IDs.",
    )

    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")

    audio_files = sorted(
        [
            p for p in audio_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]
    )

    if not audio_files:
        raise ValueError(f"No supported audio files found in {audio_dir}")

    rows = []

    for idx, path in enumerate(audio_files):
        track_id = f"{args.prefix}_{idx:05d}"
        duration = get_audio_duration(path)

        rows.append(
            {
                "track_id": track_id,
                "title": path.stem,
                "artist": "",
                "filename": path.name,
                "path": str(path),
                "duration": duration,
            }
        )

        print(f"{track_id}: {path.name} ({duration:.2f}s)")

    tracks = pd.DataFrame(rows)

    out_path = output_dir / "tracks.csv"
    tracks.to_csv(out_path, index=False)

    print(f"\nIndexed {len(tracks)} audio files.")
    print(f"Saved tracks CSV to {out_path}")
    print(tracks.head())


if __name__ == "__main__":
    main()