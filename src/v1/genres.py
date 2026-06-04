import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
AUDIO_DIR = RAW_DIR / "fma_small"
METADATA_PATH = RAW_DIR / "fma_metadata" / "tracks.csv"

def fma_track_path(track_id: int) -> Path:
    tid = f"{track_id:06d}"
    return AUDIO_DIR / tid[:3] / f"{tid}.mp3"

tracks = pd.read_csv(METADATA_PATH, index_col=0, header=[0, 1])

rows = []

for track_id, row in tracks.iterrows():
    path = fma_track_path(int(track_id))

    if not path.exists():
        continue

    rows.append({
        "track_id": int(track_id),
        "title": row.get(("track", "title"), None),
        "artist": row.get(("artist", "name"), None),
        "genre_top": row.get(("track", "genre_top"), None),
        "path": str(path),
    })

df = pd.DataFrame(rows)

print(df["genre_top"].value_counts(dropna=False))
print(df["genre_top"].dropna().unique())