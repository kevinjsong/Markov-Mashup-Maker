from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class RewardModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_feature_table_for_scoring(candidates, config):
    numeric_cols = config["feature_cols_numeric"]
    categorical_cols = config["categorical_cols"]
    expanded_feature_cols = config["expanded_feature_cols"]

    missing_numeric = [c for c in numeric_cols if c not in candidates.columns]
    missing_categorical = [c for c in categorical_cols if c not in candidates.columns]

    if missing_numeric:
        raise KeyError(f"Missing numeric columns in candidates: {missing_numeric}")

    if missing_categorical:
        raise KeyError(f"Missing categorical columns in candidates: {missing_categorical}")

    numeric = candidates[numeric_cols].astype(float).copy()

    categorical = pd.get_dummies(
        candidates[categorical_cols].fillna("unknown").astype(str),
        prefix=categorical_cols,
    )

    feature_table = pd.concat([numeric, categorical], axis=1)

    # Align columns to training-time expanded feature set.
    for col in expanded_feature_cols:
        if col not in feature_table.columns:
            feature_table[col] = 0.0

    feature_table = feature_table[expanded_feature_cols].copy()

    return feature_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates_csv",
        default="data/training_processed/transition_candidates.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="data/training_processed/transition_candidates_scored.csv",
    )
    parser.add_argument(
        "--model_dir",
        default="models",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    with open(model_dir / "reward_model_config.json", "r") as f:
        config = json.load(f)

    scaler = joblib.load(model_dir / "reward_scaler.pkl")

    candidates = pd.read_csv(args.candidates_csv)

    feature_table = build_feature_table_for_scoring(candidates, config)

    x = feature_table.astype(float).values.astype(np.float32)
    x = scaler.transform(x).astype(np.float32)

    model = RewardModel(input_dim=config["input_dim"])
    model.load_state_dict(torch.load(model_dir / "reward_model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        reward_scores = model(torch.tensor(x)).numpy()

    scored = candidates.copy()
    scored["reward_score"] = reward_scores

    scored = scored.sort_values("reward_score", ascending=False)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)

    print(f"Saved scored candidates to {out_path}")

    preview_cols = [
        "transition_id",
        "source_section_id",
        "target_section_id",
        "source_exit_key",
        "source_exit_mode",
        "target_entry_key",
        "target_entry_mode",
        "interval_name",
        "harmonic_category",
        "theory_prior_score",
        "baseline_score",
        "reward_score",
    ]

    existing = [c for c in preview_cols if c in scored.columns]
    print(scored[existing].head(20))


if __name__ == "__main__":
    main()