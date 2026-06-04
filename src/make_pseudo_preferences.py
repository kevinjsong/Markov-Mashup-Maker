from pathlib import Path
import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates_csv",
        default="data/training_processed/transition_candidates.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="data/training_processed/pseudo_preferences.csv",
    )
    parser.add_argument("--max_pairs", type=int, default=2000)
    parser.add_argument("--score_margin", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    candidates = pd.read_csv(args.candidates_csv)
    candidates = candidates.sort_values("baseline_score", ascending=False)

    rows = []
    pair_counter = 0

    for source_section_id, group in candidates.groupby("source_section_id"):
        group = group.sort_values("baseline_score", ascending=False)

        if len(group) < 4:
            continue

        high = group.head(min(5, len(group)))
        low = group.tail(min(10, len(group)))

        for _, high_row in high.iterrows():
            possible_low = low[
                high_row["baseline_score"] - low["baseline_score"] >= args.score_margin
            ]

            if possible_low.empty:
                continue

            low_row = possible_low.sample(
                n=1,
                random_state=int(rng.integers(0, 1_000_000)),
            ).iloc[0]

            rows.append(
                {
                    "preference_id": f"PSEUDO{pair_counter:07d}",
                    "comparison_id": f"PSEUDOCMP{pair_counter:07d}",
                    "context": "general",
                    "transition_a": high_row["transition_id"],
                    "transition_b": low_row["transition_id"],
                    "winner": high_row["transition_id"],
                    "label_type": "pseudo_pairwise_preference",
                    "source": "baseline_score_margin",
                    "baseline_score_a": high_row["baseline_score"],
                    "baseline_score_b": low_row["baseline_score"],
                    "score_margin": high_row["baseline_score"] - low_row["baseline_score"],
                }
            )

            pair_counter += 1

            if len(rows) >= args.max_pairs:
                break

        if len(rows) >= args.max_pairs:
            break

    pseudo = pd.DataFrame(rows)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pseudo.to_csv(out_path, index=False)

    print(f"Saved {len(pseudo)} pseudo-preferences to {out_path}")

    if len(pseudo) > 0:
        print(pseudo.head())
        print("\nScore margin summary:")
        print(pseudo["score_margin"].describe())


if __name__ == "__main__":
    main()