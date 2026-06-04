# Rubric Question Responses

## 1. Problem & Insight

I built the Markov Mashup Maker after being inspired by my a cappella group. I had to write a medley/mashup (a song that comprises snippets of other songs, usually to a specific theme). But, planning a medley was very difficult because of all the different factors to consider - song order, key relationships, modulations, and transitions - and music taste is so subjective. So, I felt that AI could step into this problem space to. learn a reward that may be difficult to explicitly define.

My main approach was to model a medley as a **Markovian transition-planning problem**:

- songs/sections = states
- possible next songs = actions
- handoffs = transition edges
- transition quality = learned reward

## 2. Execution & Technical Work

I built a functional Streamlit app that supports:

- uploading local audio files
- indexing tracks
- detecting usable song sections
- extracting musical features
- ranking best pairwise transitions
- generating full mashup routes
- explaining transition choices
- rendering audio previews

Technical components:

- audio feature extraction -> featurization: key, mode, chroma, RMS energy, MFCC/timbre, entry/exit features
- translating music theory harmony rules into a prior over transition types: interval, circle-of-fifths distance, same-key, relative major/minor, chromatic mediant, tritone, distant categories
- Bradley-Terry-style pairwise reward model: `P(A preferred over B) = sigmoid(r(A) - r(B))`
- epsilon-style preference sampling for safe vs exploratory transitions
- beam search for full route planning
- pitch-shift-aware scoring with original-key bonus and pitch-shift penalty
- transition-level audio preview rendering

I used the FMA dataset as a source of real audio for training/development candidates. I did not rely on FMA-only metadata; the model uses audio-derived features to ensure that it can extract data from user-uploaded MP3s.

## 3. Evaluation & Evidence

I evaluated the system through:

- confidence interval for audio key extraction
- manual pairwise preference collection
- frontend testing on uploaded MP3s, listening & validation of rendered transition previews

The music-theory prior was part of the baseline: same-key, circle-of-fifths, and relative major/minor transitions were initially scored as safer, while chromatic mediants, half-step shifts, tritones, and distant transitions were treated as riskier/more adventurous.

For evaluation, to ensure that learned reward wasn't simply copying this harmonic prior, I gauged reward and baseline correlation by harmonic category, and reduced the influence of pseudo-preferences + collected more human pairwise labels.

A good output should usually preserve original keys, use "safe" harmonic transitions when appropriate, but still propose creative transitions or pitch-shift bridges when they improve flow.

## 4. Communication & Presentation

The app has a clear workflow:

`Add songs → Index songs → Detect sections → Analyze features → Generate results → Render previews`

(Index songs through Analyze features were all visible for a more technical user and to show my process, but for a true user facing product, this would all likely be concealed in the backend.)

The product has two modes:

- **Best pairwise transitions**: discover strong handoffs across the corpus without choosing a starting song.
- **Full mashup routes**: choose a starting song (a la Markov chain) and generate full route options.

The output includes explanation fields: original key, realized key, pitch shift, interval, harmonic category, reward score, and audio preview. This makes the system understandable instead of purely black-box.

## 5. Process, Integrity & Disclosure

I went through many product iterations, including the following features, some of which were discontinued:

- removed explicit mood/context words to contextualize the mashup -> used starting song/current musical state as context instead
- used key-change sectioning instead of fragile chorus/bridge detection
- separated pairwise discovery from full-route planning
- separated **discovery goal** from **transposition style**
- added an a cappella/split-song option to avoid using the ending of a song as the beginning of a transition

AI tools were used for most code drafting, debugging, and documentation organization.
I designed the pipeline and training infrastructure + reward function, music theory / harmony prior encodings, product decisions, collected pairwise preferences, fine-tuned hyperparameters, and iterated on AI-drafted code.

## 6. Use Cases & Impact

Potential users include:

- a cappella arrangers!
- DJs
- musicians
- casual users who don't know music theory but still want to put together mashups and playlists

The tool makes music-theory-informed transition planning simpler and more accessible. It does not replace the musician; it gives explainable options that a human can listen to, revise, and build on. This was demonstrated in my demo: the intention was to get snapshots of potential transitions that are already "filtered" for harmonic congruence and learned user preferences.

## 7. Limitations & Future Work

Current limitations:

- key detection can be wrong
- key-change-based sectioning is usually more granular than bridge, chorus, etc.
- reward model is trained on FMA-derived candidates and my own preferences --> this would differ from person to person!
- previews are crossfaded prototypes, not professional beat-matched audio
- no DAW export, notation export, or tempo warping yet

Future work:

- more pairwise labels from multiple musicians (currently have 50 self-demonstrated pairwise preferences)
- active learning for better comparison selection
- beat-aligned rendering (for audio renders) and tempo warping (for both audio renders and potential transitions)
- improved song section detection
