from pathlib import Path
import argparse
import re

import pandas as pd

from config import MTG_METADATA_REPO, TRAINING_PROCESSED_DIR


def find_tsv_files(root: Path):
    return sorted(root.rglob("*.tsv"))


def choose_tsv(files, subset_keyword: str, prefer_unsplit: bool = True):
    subset_keyword = subset_keyword.lower()

    matches = [
        f for f in files
        if subset_keyword in str(f).lower()
    ]

    if not matches:
        print("Available TSV files:")
        for f in files[:100]:
            print(f"  {f}")
        raise FileNotFoundError(
            f"No TSV files found matching keyword: {subset_keyword}"
        )

    # Prefer the full top-level file like data/autotagging_moodtheme.tsv
    # over split files, unless requested otherwise.
    if prefer_unsplit:
        unsplit = [
            f for f in matches
            if "splits" not in str(f).lower()
        ]
        if unsplit:
            chosen = unsplit[0]
        else:
            chosen = matches[0]
    else:
        chosen = matches[0]

    print("Matched TSV files:")
    for i, f in enumerate(matches):
        marker = "  <-- using this" if f == chosen else ""
        print(f"[{i}] {f}{marker}")

    return chosen


def parse_mtg_tsv(tsv_path: Path) -> pd.DataFrame:
    """
    MTG-Jamendo annotation files have variable-length tag columns.

    Typical row:
      track_id<TAB>artist_id<TAB>album_id<TAB>path<TAB>duration<TAB>tag1<TAB>tag2...

    Because each track can have a different number of tags, pandas.read_csv()
    fails unless we parse manually.
    """
    rows = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 5:
                print(f"Skipping malformed line {line_num}: {line}")
                continue

            track_id = parts[0]
            artist_id = parts[1]
            album_id = parts[2]
            audio_path = parts[3]
            duration = parts[4]
            tags = parts[5:]

            rows.append(
                {
                    "source_track_id": track_id,
                    "artist_id": artist_id,
                    "album_id": album_id,
                    "source_path": audio_path,
                    "duration": duration,
                    "tags": tags,
                    "tags_raw": "|".join(tags),
                }
            )

    df = pd.DataFrame(rows)

    print(f"Loaded {tsv_path}")
    print(f"Shape: {df.shape}")
    print(df.head())

    return df


def clean_track_id(raw_track_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\\-]", "_", str(raw_track_id))


def normalize_tracks(df: pd.DataFrame, audio_root: Path | None = None) -> pd.DataFrame:
    """
    Convert MTG-Jamendo rows into our standard tracks.csv schema.

    Output:
      track_id,title,path,duration,tags_raw,source_dataset,source_track_id,source_path

    If local audio exists, path points to it.
    If audio is not downloaded yet, path remains empty.
    """
    rows = []

    for _, row in df.iterrows():
        source_track_id = str(row["source_track_id"])
        track_id = f"MTG_{clean_track_id(source_track_id)}"

        source_path = str(row["source_path"])
        local_path = ""

        if audio_root is not None and source_path:
            candidate = audio_root / source_path
            if candidate.exists():
                local_path = str(candidate)

        rows.append(
            {
                "track_id": track_id,
                "title": Path(source_path).stem if source_path else track_id,
                "path": local_path,
                "duration": row["duration"],
                "tags_raw": row["tags_raw"],
                "source_dataset": "mtg_jamendo",
                "source_track_id": source_track_id,
                "source_path": source_path,
            }
        )

    tracks = pd.DataFrame(rows)

    # Try to clean duration to numeric.
    tracks["duration"] = pd.to_numeric(tracks["duration"], errors="coerce")

    return tracks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_keyword", default="moodtheme")
    parser.add_argument("--metadata_repo", default=str(MTG_METADATA_REPO))
    parser.add_argument(
        "--output_csv",
        default=str(TRAINING_PROCESSED_DIR / "tracks_mtg_metadata.csv"),
    )
    parser.add_argument("--audio_root", default=None)

    args = parser.parse_args()

    repo = Path(args.metadata_repo)
    if not repo.exists():
        raise FileNotFoundError(f"MTG metadata repo not found: {repo}")

    files = find_tsv_files(repo)
    if not files:
        raise FileNotFoundError(f"No TSV files found under {repo}")

    tsv_path = choose_tsv(files, args.subset_keyword)
    df = parse_mtg_tsv(tsv_path)

    audio_root = Path(args.audio_root) if args.audio_root else None
    tracks = normalize_tracks(df, audio_root=audio_root)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(output_csv, index=False)

    print(f"\nSaved standardized MTG-Jamendo metadata to {output_csv}")
    print(tracks.head())
    print(f"Rows: {len(tracks)}")

    print("\nExample tags:")
    print(tracks["tags_raw"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()