from pathlib import Path
import random

import pandas as pd


PROCESSED_DIR = Path("data/processed")


def main(
    top_k_per_source: int = 20,
    comparisons_per_source: int = 2,
    max_comparisons: int = 100,
    seed: int = 42,
):
    """
    Create a small queue of pairwise transition comparisons for human feedback.

    Input:
        data/processed/candidate_transitions.csv

    Output:
        data/processed/feedback_queue.csv

    Each row asks:
        For a fixed source segment, which target transition is better?
    """
    random.seed(seed)

    transitions_path = PROCESSED_DIR / "candidate_transitions.csv"

    if not transitions_path.exists():
        raise FileNotFoundError(
            f"Could not find {transitions_path}. "
            "Run src/build_transitions.py first."
        )

    transitions = pd.read_csv(transitions_path)

    required_cols = {
        "transition_id",
        "source_segment_id",
        "baseline_score",
    }

    missing = required_cols - set(transitions.columns)
    if missing:
        raise ValueError(
            f"candidate_transitions.csv is missing required columns: {missing}"
        )

    # Sort highest baseline scores first.
    transitions = transitions.sort_values("baseline_score", ascending=False)

    rows = []
    comparison_counter = 0

    # Group transitions by source segment.
    # For each source segment, compare possible next segments.
    for source_segment_id, group in transitions.groupby("source_segment_id"):
        # Only use the top-k baseline candidates.
        # This avoids wasting feedback on obviously bad transitions.
        top_candidates = group.head(top_k_per_source)

        if len(top_candidates) < 2:
            continue

        transition_ids = list(top_candidates["transition_id"])

        for _ in range(comparisons_per_source):
            transition_a, transition_b = random.sample(transition_ids, 2)

            rows.append(
                {
                    "comparison_id": f"C{comparison_counter:06d}",
                    "source_segment_id": source_segment_id,
                    "transition_a": transition_a,
                    "transition_b": transition_b,
                    "status": "unlabeled",
                }
            )

            comparison_counter += 1

            if len(rows) >= max_comparisons:
                break

        if len(rows) >= max_comparisons:
            break

    queue = pd.DataFrame(rows)

    out_path = PROCESSED_DIR / "feedback_queue.csv"
    queue.to_csv(out_path, index=False)

    print(f"Saved {len(queue)} comparisons to {out_path}")

    if len(queue) > 0:
        print(queue.head(10))
    else:
        print("No comparisons created. Check candidate_transitions.csv.")


if __name__ == "__main__":
    main()