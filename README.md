# Markov Mashup Planner

Markov Mashup Planner is an AI tool for discovering and building mashups and DJ sets, where transitions between songs are just as important as the songs themselves.

The project treats mashup planning as a **Markovian transition-planning problem**: each song or section is a musical state, and the system scores possible next transitions using audio features, music theory-based priors, and a learned pairwise reward model.

The app supports two main use cases:

1. **Use the product**: upload songs, discover strong transitions, and generate mashup route options.
2. **Train your own reward model**: collect pairwise transition preferences and retrain the transition reward model.

For a direct mapping to the CS 153 project rubric, see [`RUBRIC_RESPONSES.md`](RUBRIC_RESPONSES.md).

## Setup

This project was developed with Python 3.10.

```bash
conda create -n mashup python=3.10
conda activate mashup
pip install numpy pandas scipy scikit-learn joblib torch librosa soundfile streamlit
```

## 1. Use the product

Start the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Then follow the app interface:

```text
Add songs
→ Index songs
→ Detect usable song sections
→ Analyze musical features
→ Generate results
→ Render previews
```

The app has two product modes:

### Best pairwise transitions

Use this when you want to discover strong song handoffs across a musical corpus without choosing a starting song.

The app ranks transitions between songs/sections and supports:

- Discovery goal, i.e. do we want harmonically compatible matches or harmonically interesting matches?
- Transposition style, i.e. how open are we to key changes?
- Adjustable number of transitions to show
- Option to avoid song endings as transition sources
- Audio previews of selected transitions

### Full mashup routes

Use this when you want full route examples. This mode requires:

- starting song/section
- mashup length
- transposition style

The planner uses beam search to generate route options from the selected starting state.

## 2. Do pairwise training yourself

The reward model is trained from pairwise transition preferences.

The expected training files are in:

```text
data/training_processed/
```

Key files:

```text
tracks.csv
sections.csv
transition_candidates.csv
preferences.csv
pseudo_preferences.csv
```

Collect pairwise preferences:

```bash
python src/collect_context_feedback.py \
  --tracks_csv data/training_processed/tracks.csv \
  --sections_csv data/training_processed/sections.csv \
  --candidates_csv data/training_processed/transition_candidates.csv \
  --preferences_csv data/training_processed/preferences.csv \
  --preview_dir outputs/previews \
  --max_pairs 120 \
  --epsilon 0.5
```

Generate pseudo-preferences:

```bash
python src/make_pseudo_preferences.py \
  --candidates_csv data/training_processed/transition_candidates.csv \
  --output_csv data/training_processed/pseudo_preferences.csv \
  --max_pairs 2000 \
  --score_margin 0.12
```

Retrain the reward model:

```bash
python src/train_reward_model.py \
  --candidates_csv data/training_processed/transition_candidates.csv \
  --pseudo_preferences_csv data/training_processed/pseudo_preferences.csv \
  --human_preferences_csv data/training_processed/preferences.csv \
  --model_dir models \
  --epochs 200
```

Rescore transition candidates:

```bash
python src/score_transition_candidates.py \
  --candidates_csv data/training_processed/transition_candidates.csv \
  --output_csv data/training_processed/transition_candidates_scored.csv \
  --model_dir models
```

Then restart the app:

```bash
streamlit run app/streamlit_app.py
```

## Technical summary

The system uses:

- audio-derived features: key, mode, chroma, RMS energy, MFCC/timbre, entry/exit features
- harmonic transition features: interval, circle-of-fifths distance, same-key, relative major/minor, chromatic mediant, tritone, distant categories
- music-theory prior over harmonic relationships
- pairwise Bradley-Terry-style reward learning
- epsilon-greedy-style preference sampling for safe vs exploratory comparisons
- beam search for full route planning
- pitch-shift-aware transition scoring
- transition-level audio preview rendering

## AI usage disclosure

AI tools were used for most code drafting, debugging, and documentation organization.
I designed the pipeline and training infrastructure + reward function, music theory / harmony prior encodings, product decisions, collected pairwise preferences, fine-tuned hyperparameters, and iterated on AI-drafted code.
