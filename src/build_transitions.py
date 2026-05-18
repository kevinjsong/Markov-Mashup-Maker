from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


PROCESSED_DIR = Path("data/processed")


def cosine_sim(a, b):
    a = np.asarray(a, dtype=float).reshape(1, -1)
    b = np.asarray(b, dtype=float).reshape(1, -1)
    return float(cosine_similarity(a, b)[0, 0])


def main():
    features = pd.read_csv(PROCESSED_DIR / "segment_features.csv")

    chroma_cols = [c for c in features.columns if c.startswith("chroma_")]
    mfcc_cols = [c for c in features.columns if c.startswith("mfcc_mean_")]

    rows = []
    transition_counter = 0

    print("Building candidate transitions...")

    for _, src in tqdm(features.iterrows(), total=len(features)):
        for _, tgt in features.iterrows():
            # Do not transition from a segment to itself.
            if src["segment_id"] == tgt["segment_id"]:
                continue

            # For now, skip transitions within the same original track.
            if src["track_id"] == tgt["track_id"]:
                continue

            tempo_diff = abs(float(src["tempo"]) - float(tgt["tempo"]))
            energy_diff = abs(float(src["rms_mean"]) - float(tgt["rms_mean"]))

            chroma_sim = cosine_sim(src[chroma_cols].values, tgt[chroma_cols].values)
            mfcc_sim = cosine_sim(src[mfcc_cols].values, tgt[mfcc_cols].values)

            # Convert differences to rough 0–1 compatibility scores.
            tempo_score = 1.0 - min(tempo_diff / 60.0, 1.0)
            energy_score = 1.0 - min(energy_diff / 0.2, 1.0)

            baseline_score = (
                0.30 * tempo_score
                + 0.30 * chroma_sim
                + 0.20 * mfcc_sim
                + 0.20 * energy_score
            )

            transition_id = f"T{transition_counter:06d}"
            transition_counter += 1

            rows.append(
                {
                    "transition_id": transition_id,
                    "source_segment_id": src["segment_id"],
                    "target_segment_id": tgt["segment_id"],
                    "source_track_id": src["track_id"],
                    "target_track_id": tgt["track_id"],
                    "source_position": src["segment_position"],
                    "target_position": tgt["segment_position"],
                    "tempo_diff": tempo_diff,
                    "energy_diff": energy_diff,
                    "chroma_sim": chroma_sim,
                    "mfcc_sim": mfcc_sim,
                    "baseline_score": baseline_score,
                }
            )

    transitions = pd.DataFrame(rows)
    transitions = transitions.sort_values("baseline_score", ascending=False)

    out_path = PROCESSED_DIR / "candidate_transitions.csv"
    transitions.to_csv(out_path, index=False)

    print(f"Saved {len(transitions)} transitions to {out_path}")
    print(transitions.head(10))


if __name__ == "__main__":
    main()