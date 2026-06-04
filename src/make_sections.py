from pathlib import Path
import argparse

import pandas as pd


def make_full_track_sections(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Create one section per track.

    For FMA-small, this is appropriate because each audio file is already
    roughly a 30-second clip.
    """
    rows = []

    for _, track in tracks.iterrows():
        track_id = track["track_id"]
        duration = float(track["duration"])

        rows.append(
            {
                "section_id": f"{track_id}_full",
                "track_id": track_id,
                "section_label": "full_clip",
                "start_time": 0.0,
                "end_time": round(duration, 3),
                "hook_score": 1,
                "source": "full_clip",
                "confidence": 1.0,
            }
        )

    return pd.DataFrame(rows)


def make_window_sections(
    tracks: pd.DataFrame,
    window_sec: float,
    hop_sec: float,
    min_section_sec: float,
) -> pd.DataFrame:
    """
    Create automatic fixed-window sections.

    This will be useful later for full songs / MTG-Jamendo-style data.
    """
    rows = []

    for _, track in tracks.iterrows():
        track_id = track["track_id"]
        duration = float(track["duration"])

        start = 0.0
        idx = 0

        while start < duration:
            end = min(start + window_sec, duration)

            if end - start >= min_section_sec:
                rows.append(
                    {
                        "section_id": f"{track_id}_w{idx:03d}",
                        "track_id": track_id,
                        "section_label": "auto_window",
                        "start_time": round(start, 3),
                        "end_time": round(end, 3),
                        "hook_score": 1,
                        "source": "auto_window",
                        "confidence": 0.5,
                    }
                )

            start += hop_sec
            idx += 1

    return pd.DataFrame(rows)


def make_manual_template(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Create a section annotation template for final setlist songs.
    """
    rows = []

    for _, track in tracks.iterrows():
        track_id = track["track_id"]
        title = track.get("title", track_id)
        duration = float(track["duration"])

        rows.append(
            {
                "section_id": f"{track_id}_section001",
                "track_id": track_id,
                "track_title": title,
                "section_label": "chorus",
                "start_time": 0.0,
                "end_time": round(min(30.0, duration), 3),
                "hook_score": 3,
                "source": "manual",
                "confidence": 1.0,
                "notes": "EDIT/duplicate rows; save final as sections.csv",
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument(
        "--mode",
        choices=["full", "windows", "manual_template"],
        required=True,
    )
    parser.add_argument("--window_sec", type=float, default=20.0)
    parser.add_argument("--hop_sec", type=float, default=20.0)
    parser.add_argument("--min_section_sec", type=float, default=8.0)

    args = parser.parse_args()

    tracks = pd.read_csv(args.tracks_csv)

    if args.mode == "full":
        sections = make_full_track_sections(tracks)
    elif args.mode == "windows":
        sections = make_window_sections(
            tracks,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            min_section_sec=args.min_section_sec,
        )
    elif args.mode == "manual_template":
        sections = make_manual_template(tracks)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sections.to_csv(out_path, index=False)

    print(f"Saved {len(sections)} sections to {out_path}")
    print(sections.head())


if __name__ == "__main__":
    main()