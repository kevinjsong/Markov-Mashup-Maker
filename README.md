# Markov Mashup Maker

Markov Mashup Maker is an AI tool for discovering and building mashups and DJ sets, where transitions between songs are just as important as the songs themselves.

The project treats mashup planning as a **Markovian transition-planning problem**: each song or section is a musical state, and the system scores possible next transitions using audio features, music theory-based priors, and a learned pairwise reward model.
So, you simply upload songs, discover strong transitions, and generate mashup route options.

For explicit project rubric responses, see [`RUBRIC_RESPONSES.md`](RUBRIC_RESPONSES.md).

## Setup

This project was developed with Python 3.10.

```bash
conda create -n mashup python=3.10
conda activate mashup
pip install numpy pandas scipy scikit-learn joblib torch librosa soundfile streamlit tqdm
```

## I. Using the App: Quickstart

The app can run immediately with the pretrained reward model included in:

```text
models/
├── reward_model.pt
├── reward_model_config.json
└── reward_scaler.pkl
```

To do so, start the Streamlit app:

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

Use this to explore the best potential transitions between songs across a musical corpus -- no need to specify a starting song. This mode supports:
- Discovery goal, i.e. do we want harmonically compatible matches or harmonically interesting matches?
- Transposition style, i.e. how open are we to key changes?
- Adjustable number of transitions to show
- Option to avoid song endings as transition sources
- Audio previews of selected transitions

### Full mashup routes

Use this when you want full mashup examples. This mode requires a starting song / section, mashup length, and transposition style. 
The planner then uses beam search to generate route options from the selected starting state.

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

---

## II. Training your own Markov Mashup Maker: Quickstart

Retraining is optional -- but music taste is subjective, and you may be interested in having your own AI mashup assistant! 
The raw `data/` directory is not committed. To retrain the reward model, download FMA-small and FMA metadata locally (https://github.com/mdeff/fma) and place them under:

```text
data/
└── raw/
    ├── fma_small/
    │   ├── 000/
    │   ├── 001/
    │   └── ...
    └── fma_metadata/
        └── tracks.csv
```

Then, run the v2 FMA training pipeline:

```bash
# Data pre-processing
python src/import_fma_training_subset.py \
  --metadata_csv data/raw/fma_metadata/tracks.csv \
  --audio_dir data/raw/fma_small \
  --output_dir data/training_processed \
  --genres Pop Folk \
  --max_tracks 300

python src/make_sections.py \
  --tracks_csv data/training_processed/tracks.csv \
  --output_csv data/training_processed/sections.csv \
  --mode full

python src/extract_section_features.py \
  --tracks_csv data/training_processed/tracks.csv \
  --sections_csv data/training_processed/sections.csv \
  --section_features_csv data/training_processed/section_features.csv \
  --boundary_features_csv data/training_processed/boundary_features.csv

python src/generate_transition_candidates.py \
  --section_features_csv data/training_processed/section_features.csv \
  --boundary_features_csv data/training_processed/boundary_features.csv \
  --output_csv data/training_processed/transition_candidates.csv \
  --max_candidates 30000 \
  --top_k_per_source 40 \
  --epsilon 0.20

# Initiate pairwise transition preference selection
python src/collect_context_feedback.py \
  --tracks_csv data/training_processed/tracks.csv \
  --sections_csv data/training_processed/sections.csv \
  --candidates_csv data/training_processed/transition_candidates.csv \
  --preferences_csv data/training_processed/preferences.csv \
  --preview_dir outputs/previews \
  --max_pairs 120 \
  --epsilon 0.5

# Generate pseudo-preferences from theoretical harmonic baseline
python src/make_pseudo_preferences.py \
  --candidates_csv data/training_processed/transition_candidates.csv \
  --output_csv data/training_processed/pseudo_preferences.csv \
  --max_pairs 2000 \
  --score_margin 0.12

# Train the model
python src/train_reward_model.py \
  --candidates_csv data/training_processed/transition_candidates.csv \
  --pseudo_preferences_csv data/training_processed/pseudo_preferences.csv \
  --human_preferences_csv data/training_processed/preferences.csv \
  --model_dir models \
  --epochs 200

# Score all transition candidates wit learned reward 
python src/score_transition_candidates.py \
  --candidates_csv data/training_processed/transition_candidates.csv \
  --output_csv data/training_processed/transition_candidates_scored.csv \
  --model_dir models
```

This generates:

```text
data/
└── training_processed/
    ├── tracks.csv
    ├── sections.csv
    ├── section_features.csv
    ├── boundary_features.csv
    ├── transition_candidates.csv
    ├── preferences.csv
    ├── pseudo_preferences.csv
    └── transition_candidates_scored.csv
```
