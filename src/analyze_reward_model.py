from pathlib import Path
import argparse

import pandas as pd
from sklearn.preprocessing import minmax_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scored_csv",
        default="data/training_processed/transition_candidates_scored.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="data/training_processed/reward_analysis",
    )
    parser.add_argument("--top_n", type=int, default=50)

    args = parser.parse_args()

    scored = pd.read_csv(args.scored_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = scored.copy()

    scored["baseline_norm"] = minmax_scale(scored["baseline_score"])
    scored["reward_norm"] = minmax_scale(scored["reward_score"])
    scored["reward_minus_baseline"] = scored["reward_norm"] - scored["baseline_norm"]

    print("Shape:", scored.shape)

    print("\nBaseline/reward correlation:")
    print(scored[["baseline_score", "reward_score"]].corr())

    print("\nReward by harmonic category:")
    category_summary = (
        scored.groupby("harmonic_category")
        .agg(
            count=("transition_id", "count"),
            mean_reward=("reward_score", "mean"),
            median_reward=("reward_score", "median"),
            max_reward=("reward_score", "max"),
            mean_theory_prior=("theory_prior_score", "mean"),
            mean_disagreement=("reward_minus_baseline", "mean"),
        )
        .sort_values("mean_reward", ascending=False)
    )
    print(category_summary)

    category_summary.to_csv(out_dir / "reward_by_harmonic_category.csv")

    top_reward = scored.sort_values("reward_score", ascending=False).head(args.top_n)
    top_reward.to_csv(out_dir / "top_by_reward.csv", index=False)

    top_baseline = scored.sort_values("baseline_score", ascending=False).head(args.top_n)
    top_baseline.to_csv(out_dir / "top_by_baseline.csv", index=False)

    high_model_low_theory = scored.sort_values(
        "reward_minus_baseline",
        ascending=False,
    ).head(args.top_n)
    high_model_low_theory.to_csv(out_dir / "model_likes_more_than_theory.csv", index=False)

    high_theory_low_model = scored.sort_values(
        "reward_minus_baseline",
        ascending=True,
    ).head(args.top_n)
    high_theory_low_model.to_csv(out_dir / "theory_likes_more_than_model.csv", index=False)

    print(f"\nSaved analysis CSVs to {out_dir}")


if __name__ == "__main__":
    main()