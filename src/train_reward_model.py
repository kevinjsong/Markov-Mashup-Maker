from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "raw_interval_semitones",
    "signed_interval_semitones",
    "abs_interval_semitones",
    "circle_of_fifths_distance",
    "same_tonic",
    "same_mode",
    "same_key_mode",
    "parallel_mode",
    "relative_major_to_minor",
    "relative_minor_to_major",
    "relative_major_minor",
    "perfect_fifth_relation",
    "circle_neighbor",
    "near_circle",
    "half_step_relation",
    "whole_step_relation",
    "tritone_relation",
    "chromatic_mediant_candidate",
    "theory_prior_score",
    "chroma_sim_raw",
    "chroma_sim_after_shift",
    "mfcc_sim",
    "energy_change",
    "energy_abs_diff",
    "source_exit_chroma_entropy",
    "target_entry_chroma_entropy",
    "baseline_score",
]

CATEGORICAL_COLS = [
    "interval_name",
    "harmonic_category",
    "sampling_bucket",
    "source_exit_key",
    "source_exit_mode",
    "target_entry_key",
    "target_entry_mode",
]


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


def build_feature_table(candidates: pd.DataFrame):
    missing_numeric = [c for c in FEATURE_COLS if c not in candidates.columns]
    missing_categorical = [c for c in CATEGORICAL_COLS if c not in candidates.columns]

    if missing_numeric:
        raise KeyError(f"Missing numeric feature columns: {missing_numeric}")

    if missing_categorical:
        raise KeyError(f"Missing categorical feature columns: {missing_categorical}")

    numeric = candidates[FEATURE_COLS].astype(float).copy()

    categorical = pd.get_dummies(
        candidates[CATEGORICAL_COLS].fillna("unknown").astype(str),
        prefix=CATEGORICAL_COLS,
    )

    feature_table = pd.concat([numeric, categorical], axis=1)
    feature_table.insert(0, "transition_id", candidates["transition_id"].values)

    return feature_table


def load_preferences(pseudo_csv: str | None, human_csv: str | None):
    frames = []

    if pseudo_csv and Path(pseudo_csv).exists():
        pseudo = pd.read_csv(pseudo_csv)
        pseudo = pseudo[pseudo["label_type"] == "pseudo_pairwise_preference"].copy()
        pseudo["weight"] = 0.10
        frames.append(pseudo)

    if human_csv and Path(human_csv).exists():
        human = pd.read_csv(human_csv)
        human = human[human["label_type"] == "pairwise_preference"].copy()
        human["weight"] = 1.0
        frames.append(human)

    if not frames:
        raise FileNotFoundError("No usable pseudo or human pairwise preferences found.")

    prefs = pd.concat(frames, ignore_index=True)

    required = ["transition_a", "transition_b", "winner", "weight"]
    missing = [c for c in required if c not in prefs.columns]
    if missing:
        raise KeyError(f"Preference file missing columns: {missing}")

    return prefs


def make_pairwise_arrays(feature_table, prefs):
    features_by_id = feature_table.set_index("transition_id")

    x_a = []
    x_b = []
    y = []
    weights = []

    feature_cols = [c for c in feature_table.columns if c != "transition_id"]

    for _, pref in prefs.iterrows():
        ta = pref["transition_a"]
        tb = pref["transition_b"]
        winner = pref["winner"]

        if ta not in features_by_id.index or tb not in features_by_id.index:
            continue

        fa = features_by_id.loc[ta, feature_cols].astype(float).values
        fb = features_by_id.loc[tb, feature_cols].astype(float).values

        x_a.append(fa)
        x_b.append(fb)

        # y=1 means A wins; y=0 means B wins.
        y.append(1.0 if winner == ta else 0.0)
        weights.append(float(pref.get("weight", 1.0)))

    if not x_a:
        raise ValueError("No valid preference pairs after matching transition IDs.")

    return (
        np.asarray(x_a, dtype=np.float32),
        np.asarray(x_b, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        feature_cols,
    )


def train_model(x_a, x_b, y, weights, epochs=200, lr=1e-3):
    scaler = StandardScaler()
    scaler.fit(np.vstack([x_a, x_b]))

    x_a = scaler.transform(x_a).astype(np.float32)
    x_b = scaler.transform(x_b).astype(np.float32)

    x_a = torch.tensor(x_a)
    x_b = torch.tensor(x_b)
    y = torch.tensor(y)
    weights = torch.tensor(weights)

    model = RewardModel(input_dim=x_a.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(epochs):
        model.train()

        r_a = model(x_a)
        r_b = model(x_b)

        logits = r_a - r_b
        losses = loss_fn(logits, y)

        weighted_loss = (losses * weights).mean()

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                acc = (preds == y).float().mean().item()

                human_mask = weights >= 1.0
                if human_mask.any():
                    human_acc = (preds[human_mask] == y[human_mask]).float().mean().item()
                else:
                    human_acc = float("nan")

            print(
                f"epoch={epoch:03d} "
                f"loss={weighted_loss.item():.4f} "
                f"acc={acc:.3f} "
                f"human_acc={human_acc:.3f}"
            )

    return model, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates_csv",
        default="data/training_processed/transition_candidates.csv",
    )
    parser.add_argument(
        "--pseudo_preferences_csv",
        default="data/training_processed/pseudo_preferences.csv",
    )
    parser.add_argument(
        "--human_preferences_csv",
        default="data/training_processed/preferences.csv",
    )
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates_csv)
    feature_table = build_feature_table(candidates)

    prefs = load_preferences(
        pseudo_csv=args.pseudo_preferences_csv,
        human_csv=args.human_preferences_csv,
    )

    print(f"Loaded {len(candidates)} candidates")
    print(f"Loaded {len(prefs)} training preference pairs")
    print(prefs["label_type"].value_counts())

    x_a, x_b, y, weights, feature_cols = make_pairwise_arrays(feature_table, prefs)

    print(f"Training pairs after matching IDs: {len(y)}")
    print(f"Input dimension before scaling: {x_a.shape[1]}")

    model, scaler = train_model(
        x_a=x_a,
        x_b=x_b,
        y=y,
        weights=weights,
        epochs=args.epochs,
        lr=args.lr,
    )

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_dir / "reward_model.pt")
    joblib.dump(scaler, model_dir / "reward_scaler.pkl")

    config = {
        "feature_cols_numeric": FEATURE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "expanded_feature_cols": feature_cols,
        "input_dim": x_a.shape[1],
    }

    with open(model_dir / "reward_model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved model to {model_dir / 'reward_model.pt'}")
    print(f"Saved scaler to {model_dir / 'reward_scaler.pkl'}")
    print(f"Saved config to {model_dir / 'reward_model_config.json'}")


if __name__ == "__main__":
    main()