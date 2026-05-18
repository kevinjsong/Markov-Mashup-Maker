# CS 153 Final Project: A Cappella Mashup Creator

My project explores a human feedback-driven system for reward learning to recommend mashup transitions between songs to qualitatively better fit a user's taste.

Current version: Processes a music dataset (I used Free Music Archive, or FMA) into audio features, builds candidate transitions between clips, ranks those transitions with a heuristic baseline score, and sets up a preference-labeling workflow for future reward-model training.

## Current Pipeline

1. Load FMA-small audio metadata.
2. Filter tracks to selected genres.
3. Treat each 30-second FMA clip as one musical segment.
4. Extract segment-level audio features:
   - tempo
   - chroma
   - MFCCs
   - RMS energy
   - spectral centroid
5. Build candidate transitions between clips.
6. Score transitions using a heuristic compatibility function.
7. Create a feedback queue for pairwise human preference labeling.
8. Collect preference labels by playing audio previews directly from the terminal.

## Repo Structure

```text
src/
  process_fma.py
  build_transitions.py
  make_feedback_queue.py
  collect_preferences.py

data/
  processed/
    tracks.csv
    segments.csv
    segment_features.csv
    candidate_transitions.csv
    feedback_queue.csv
```
