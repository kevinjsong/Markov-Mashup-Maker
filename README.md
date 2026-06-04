# Markov Mashup Maker

Markov Mashup Maker is an AI tool for discovering and building mashups and DJ sets, where transitions between songs are just as important as the songs themselves.

The project treats mashup planning as a **Markovian transition-planning problem**: each song or section is a musical state, and the system scores possible next transitions using audio features, music theory-based priors, and a learned pairwise reward model.
So, you simply upload songs, discover strong transitions, and generate mashup route options.

For a direct mapping to the project rubric, see [`RUBRIC_RESPONSES.md`](RUBRIC_RESPONSES.md).

## Setup

This project was developed with Python 3.10.

```bash
conda create -n mashup python=3.10
conda activate mashup
pip install numpy pandas scipy scikit-learn joblib torch librosa soundfile streamlit tqdm
```

## Quickstart Instructions

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
