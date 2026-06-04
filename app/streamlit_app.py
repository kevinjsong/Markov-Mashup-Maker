from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

UPLOAD_AUDIO_DIR = PROJECT_ROOT / "data" / "user_uploads" / "audio"
UPLOAD_PROCESSED_DIR = PROJECT_ROOT / "data" / "user_uploads" / "processed"
OUTPUT_PREVIEW_DIR = PROJECT_ROOT / "outputs" / "user_previews"

TRACKS_CSV = UPLOAD_PROCESSED_DIR / "tracks.csv"
SECTIONS_CSV = UPLOAD_PROCESSED_DIR / "sections.csv"
SECTION_FEATURES_CSV = UPLOAD_PROCESSED_DIR / "section_features.csv"
BOUNDARY_FEATURES_CSV = UPLOAD_PROCESSED_DIR / "boundary_features.csv"

PLANS_CSV = UPLOAD_PROCESSED_DIR / "mashup_plans_with_bridges.csv"
PLANS_JSON = UPLOAD_PROCESSED_DIR / "mashup_plans_with_bridges.json"

PAIRWISE_SMOOTH_CSV = UPLOAD_PROCESSED_DIR / "best_pairwise_transitions_smooth.csv"
PAIRWISE_CREATIVE_CSV = UPLOAD_PROCESSED_DIR / "best_pairwise_transitions_creative.csv"


def run_cmd(cmd):
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        with st.expander("Command output", expanded=False):
            st.code(result.stdout)

    if result.returncode != 0:
        st.error("Command failed.")
        st.code(result.stderr)
        st.stop()

    return result


def save_uploaded_files(uploaded_files, clear_existing=False):
    UPLOAD_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if clear_existing:
        for p in UPLOAD_AUDIO_DIR.iterdir():
            if p.is_file():
                p.unlink()

    saved_paths = []

    for file in uploaded_files:
        out_path = UPLOAD_AUDIO_DIR / file.name

        with open(out_path, "wb") as f:
            f.write(file.getbuffer())

        saved_paths.append(out_path)

    return saved_paths


def load_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None


def clear_processed_outputs():
    UPLOAD_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for path in [
        TRACKS_CSV,
        SECTIONS_CSV,
        SECTION_FEATURES_CSV,
        BOUNDARY_FEATURES_CSV,
        PLANS_CSV,
        PLANS_JSON,
        PAIRWISE_SMOOTH_CSV,
        PAIRWISE_CREATIVE_CSV,
        UPLOAD_PROCESSED_DIR / "selected_pairwise_transition_plan.csv",
    ]:
        if path.exists():
            path.unlink()


def get_transposition_settings(style):
    if style == "Conservative":
        return {
            "max_abs_pitch_shift": 2,
            "original_key_bonus": 0.14,
            "pitch_shift_penalty": 0.075,
        }

    if style == "Adventurous":
        return {
            "max_abs_pitch_shift": 5,
            "original_key_bonus": 0.00,
            "pitch_shift_penalty": 0.012,
        }

    return {
        "max_abs_pitch_shift": 4,
        "original_key_bonus": 0.08,
        "pitch_shift_penalty": 0.035,
    }


def get_discovery_settings(transposition_style, discovery_goal_arg):
    if discovery_goal_arg == "smooth":
        if transposition_style == "Adventurous":
            return {
                "discovery_weight": 0.20,
                "reward_weight": 0.65,
                "theory_weight": 0.12,
                "baseline_weight": 0.08,
            }

        return {
            "discovery_weight": 0.20,
            "reward_weight": 0.65,
            "theory_weight": 0.15,
            "baseline_weight": 0.10,
        }

    # Creative / dissonant discovery.
    if transposition_style == "Adventurous":
        return {
            "discovery_weight": 0.50,
            "reward_weight": 0.55,
            "theory_weight": 0.08,
            "baseline_weight": 0.06,
        }

    if transposition_style == "Conservative":
        return {
            "discovery_weight": 0.25,
            "reward_weight": 0.65,
            "theory_weight": 0.12,
            "baseline_weight": 0.08,
        }

    return {
        "discovery_weight": 0.35,
        "reward_weight": 0.60,
        "theory_weight": 0.10,
        "baseline_weight": 0.08,
    }


def make_pitch_shift_text(pitch_shift):
    if pd.isna(pitch_shift):
        return "original key"

    if str(pitch_shift).strip() == "":
        return "original key"

    pitch_shift = int(float(pitch_shift))

    if pitch_shift == 0:
        return "original key"

    return f"shifted {pitch_shift:+} semitones"


def make_transition_explanation(row):
    category = row.get("incoming_harmonic_category", row.get("harmonic_category", "unknown"))
    explanation_parts = []

    if category == "same_key":
        explanation_parts.append(
            "This is a stable same-key transition, so the harmonic handoff should feel smooth."
        )
    elif category == "circle_neighbor":
        explanation_parts.append(
            "This follows a closely related circle-of-fifths movement, which often gives a strong but natural sense of motion."
        )
    elif category in {"relative_major_to_minor", "relative_minor_to_major"}:
        explanation_parts.append(
            "This uses a relative major/minor relationship, which can preserve continuity while shifting mood."
        )
    elif category == "chromatic_mediant":
        explanation_parts.append(
            "This is a more colorful chromatic-mediant-style move, which can feel dramatic or cinematic."
        )
    elif category == "near_circle":
        explanation_parts.append(
            "This is near the source key on the circle of fifths, so it is moderately related while still creating movement."
        )
    elif category in {"half_step_shift", "tritone", "distant_other"}:
        explanation_parts.append(
            "This is a riskier harmonic move that may create more tension or surprise."
        )
    elif category == "whole_step_shift":
        explanation_parts.append(
            "This moves by a whole step, which can create a noticeable but still manageable lift or drop."
        )

    preserves_original = row.get("target_preserves_original_key", None)

    if preserves_original is not None and not pd.isna(preserves_original):
        if str(preserves_original).strip() != "":
            if int(float(preserves_original)) == 1:
                explanation_parts.append("The target section stays in its detected original key.")
            else:
                explanation_parts.append(
                    "The planner proposes a pitch shift as a bridge to improve harmonic flow."
                )

    reward = row.get("incoming_reward_score", row.get("reward_score", None))

    if reward is not None and not pd.isna(reward):
        if str(reward).strip() != "":
            explanation_parts.append(
                f"The learned reward score for this transition is {float(reward):.3f}."
            )

    if not explanation_parts:
        return (
            "This transition was selected based on learned reward, harmonic fit, "
            "energy/timbre compatibility, and pitch-shift cost."
        )

    return " ".join(explanation_parts)


def safe_display_value(value, fallback=""):
    if value is None:
        return fallback
    if pd.isna(value):
        return fallback
    if str(value).strip() == "":
        return fallback
    return str(value)


def make_section_display_table(section_features, tracks):
    display = section_features.copy()

    if tracks is not None and "track_id" in tracks.columns:
        track_cols = [
            c for c in ["track_id", "title", "artist", "path"]
            if c in tracks.columns
        ]
        display = display.merge(tracks[track_cols], on="track_id", how="left")

    def make_name(row):
        title = safe_display_value(row.get("title", ""), fallback="")
        artist = safe_display_value(row.get("artist", ""), fallback="")
        section_label = safe_display_value(row.get("section_label", ""), fallback="section")
        key = safe_display_value(row.get("key", ""), fallback="?")
        mode = safe_display_value(row.get("mode", ""), fallback="?")
        section_id = safe_display_value(row.get("section_id", ""), fallback="unknown_section")
        path = safe_display_value(row.get("path", ""), fallback="")

        start = row.get("start_time", None)
        end = row.get("end_time", None)

        if not title:
            if path:
                title = Path(path).stem
            else:
                title = section_id

        title_part = title
        if artist:
            title_part = f"{title_part} — {artist}"

        if start is not None and end is not None and not pd.isna(start) and not pd.isna(end):
            time_part = f"{float(start):.1f}s–{float(end):.1f}s"
        else:
            time_part = "unknown time"

        details = f"{section_label} | {key} {mode} | {time_part}"
        return f"{title_part} ({details})"

    display["display_name"] = display.apply(make_name, axis=1)
    return display


def merge_tracks_for_display(rows, tracks, source_col=None, target_col=None):
    out = rows.copy()

    if tracks is None or "track_id" not in tracks.columns:
        return out

    track_cols = [
        c for c in ["track_id", "title", "artist", "path"]
        if c in tracks.columns
    ]

    if source_col and source_col in out.columns:
        source_tracks = tracks[track_cols].rename(
            columns={
                "track_id": source_col,
                "title": "source_title",
                "artist": "source_artist",
                "path": "source_path",
            }
        )
        out = out.merge(source_tracks, on=source_col, how="left")

    if target_col and target_col in out.columns:
        target_tracks = tracks[track_cols].rename(
            columns={
                "track_id": target_col,
                "title": "target_title",
                "artist": "target_artist",
                "path": "target_path",
            }
        )
        out = out.merge(target_tracks, on=target_col, how="left")

    return out


def write_pairwise_transition_as_plan_csv(row, output_csv):
    """
    Create a tiny 2-row pseudo-plan so render_plan_previews.py can render
    a transition preview for a selected pairwise edge.
    """
    plan_id = "PAIRWISE_SELECTED"

    source_section_id = row["source_section_id"]
    target_section_id = row["target_section_id"]

    source_track_id = row["source_track_id"]
    target_track_id = row["target_track_id"]

    source_key = f"{row.get('source_exit_key', '')} {row.get('source_exit_mode', '')}".strip()
    target_key = f"{row.get('target_realized_key', '')} {row.get('target_entry_mode', '')}".strip()

    pitch_shift = row.get("target_pitch_shift", 0)

    rows = [
        {
            "plan_id": plan_id,
            "plan_score": row.get("final_edge_score", 0),
            "step": 0,
            "section_id": source_section_id,
            "track_id": source_track_id,
            "original_key": "",
            "realized_key": "",
            "pitch_shift": "",
            "incoming_transition_id": "",
            "incoming_from_key": "",
            "incoming_to_key": "",
            "incoming_harmonic_category": "",
            "incoming_interval_name": "",
            "incoming_reward_score": "",
            "target_preserves_original_key": "",
        },
        {
            "plan_id": plan_id,
            "plan_score": row.get("final_edge_score", 0),
            "step": 1,
            "section_id": target_section_id,
            "track_id": target_track_id,
            "original_key": row.get("target_original_key", ""),
            "realized_key": row.get("target_realized_key", ""),
            "pitch_shift": pitch_shift,
            "incoming_transition_id": row.get("transition_id", ""),
            "incoming_from_key": source_key,
            "incoming_to_key": target_key,
            "incoming_harmonic_category": row.get("harmonic_category", ""),
            "incoming_interval_name": row.get("interval_name", ""),
            "incoming_reward_score": row.get("reward_score", ""),
            "target_preserves_original_key": row.get("target_preserves_original_key", ""),
        },
    ]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def main():
    st.set_page_config(
        page_title="Markov Mashup Maker",
        layout="wide",
    )

    st.title("Markov Mashup Maker")
    st.caption(
        "Plan full mashup routes or discover strong pairwise transitions using key, energy, timbre, and learned reward."
    )

    st.info(
        """
        This app applies a learned transition reward model to your uploaded songs.
        It can either build full Markovian mashup routes from a chosen starting song, or rank the best pairwise
        transitions across the whole corpus.
        """
    )

    st.sidebar.header("Mashup controls")

    planning_mode = st.sidebar.radio(
        "What do you want to generate?",
        ["Full mashup routes", "Best pairwise transitions"],
        index=0,
        help=(
            "Full mashup routes builds complete multi-step routes from a selected start. "
            "Best pairwise transitions ranks strong handoffs anywhere in the corpus."
        ),
    )

    if planning_mode == "Full mashup routes":
        path_length = st.sidebar.slider(
            "Mashup length",
            3,
            8,
            5,
            help="How many songs/sections to include in the generated route.",
        )
        pairwise_top_n = 50
    else:
        path_length = 2
        pairwise_top_n = st.sidebar.slider(
            "Number of transitions to show",
            10,
            100,
            25,
            help="How many top-ranked pairwise transitions to return.",
        )

    transposition_style = st.sidebar.radio(
        "Transposition style",
        ["Conservative", "Balanced", "Adventurous"],
        index=1,
        help=(
            "Conservative strongly prefers original keys. "
            "Balanced allows useful bridge transpositions. "
            "Adventurous is more open to bold key changes and larger pitch shifts."
        ),
    )

    transposition_settings = get_transposition_settings(transposition_style)

    with st.sidebar.expander("Advanced search settings"):
        st.write("These control search/ranking. The defaults are usually fine.")

        beam_size = st.slider(
            "Beam size",
            5,
            50,
            10,
            help="For full-route planning: how many partial plans are kept at each search step.",
        )

        initial_sections = st.slider(
            "Initial sections",
            5,
            50,
            20,
            help="Fallback setting for route planning.",
        )

        candidate_edges_per_step = st.slider(
            "Candidate transitions per step",
            50,
            500,
            120,
            help="For full-route planning: how many next-transition candidates are scored at each step.",
        )

        max_candidates_to_score = st.slider(
            "Pairwise candidates to score",
            500,
            10000,
            5000,
            step=500,
            help="For pairwise discovery: more candidates gives broader search but can be slower.",
        )

    with st.sidebar.expander("How planning works"):
        st.write(
            """
            **Full mashup routes** uses beam search from a selected starting section.
            Each next choice depends on the current song/key/audio state.

            **Best pairwise transitions** ranks individual transition edges across the uploaded corpus.
            This is useful when you do not yet know the full route and just want to discover strong handoffs.

            Transposition style affects both modes by controlling how strongly the system prefers original keys
            versus allowing more adventurous pitch-shift bridges.
            """
        )

    st.header("1. Add songs")

    audio_source = st.radio(
        "Audio source",
        ["Upload files", "Use existing local folder"],
        index=0,
        horizontal=True,
    )

    if audio_source == "Upload files":
        uploaded_files = st.file_uploader(
            "Upload audio files",
            type=["mp3", "wav", "m4a", "flac", "ogg", "aiff", "aif"],
            accept_multiple_files=True,
        )

        clear_existing_uploads = st.checkbox(
            "Clear existing uploaded songs before saving",
            value=False,
        )

        if uploaded_files:
            if st.button("Save uploaded songs"):
                saved = save_uploaded_files(
                    uploaded_files,
                    clear_existing=clear_existing_uploads,
                )
                st.success(f"Saved {len(saved)} files.")
                for p in saved:
                    st.write(p.relative_to(PROJECT_ROOT))

    else:
        st.write("Put audio files in this folder, then click **Index songs** below:")
        st.code(str(UPLOAD_AUDIO_DIR.relative_to(PROJECT_ROOT)))
        UPLOAD_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

        current_audio_files = [
            p for p in UPLOAD_AUDIO_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in {
                ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aiff", ".aif"
            }
        ]

        st.write(f"Found {len(current_audio_files)} audio files.")
        for p in current_audio_files[:20]:
            st.write(p.name)

    st.divider()

    st.header("2. Index songs")

    st.write(
        "Indexing reads the uploaded audio files and creates a track table with file paths and durations."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Index songs"):
            UPLOAD_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

            run_cmd([
                sys.executable,
                "src/index_local_audio.py",
                "--audio_dir",
                str(UPLOAD_AUDIO_DIR.relative_to(PROJECT_ROOT)),
                "--output_dir",
                str(UPLOAD_PROCESSED_DIR.relative_to(PROJECT_ROOT)),
                "--prefix",
                "USER",
            ])

            st.success("Indexed songs.")

    with col2:
        if st.button("Clear processed outputs"):
            clear_processed_outputs()
            st.success("Cleared processed CSV outputs.")

    tracks = load_csv_if_exists(TRACKS_CSV)

    if tracks is not None:
        st.subheader("Indexed tracks")
        st.dataframe(tracks, use_container_width=True)

    st.divider()

    st.header("3. Choose sectioning mode")

    section_mode = st.radio(
        "Section creation mode",
        [
            "key_change_sections",
            "full_song_dj_mode",
        ],
        index=0,
        format_func=lambda x: {
            "key_change_sections": "Key-change sections",
            "full_song_dj_mode": "Full-song / DJ mode",
        }[x],
        horizontal=False,
        help=(
            "Key-change sections split songs when estimated key/mode changes persistently. "
            "Full-song mode treats each uploaded song as one unit."
        ),
    )

    st.info(
        """
        Recommended: **Key-change sections**.  
        This splits a song only when its estimated key/mode changes persistently, which is useful
        for bridges, modulations, or outros that move to a new harmonic region.

        Use **Full-song / DJ mode** when you want end-to-end song sequencing.
        """
    )

    if st.button("Create sections"):
        if not TRACKS_CSV.exists():
            st.error("Index songs first.")
            st.stop()

        if section_mode == "key_change_sections":
            cmd = [
                sys.executable,
                "src/make_key_change_sections.py",
                "--tracks_csv",
                str(TRACKS_CSV.relative_to(PROJECT_ROOT)),
                "--output_csv",
                str(SECTIONS_CSV.relative_to(PROJECT_ROOT)),
                "--window_sec",
                "12",
                "--hop_sec",
                "6",
                "--min_section_sec",
                "20",
                "--min_persist_windows",
                "2",
            ]

        elif section_mode == "full_song_dj_mode":
            cmd = [
                sys.executable,
                "src/make_sections.py",
                "--tracks_csv",
                str(TRACKS_CSV.relative_to(PROJECT_ROOT)),
                "--output_csv",
                str(SECTIONS_CSV.relative_to(PROJECT_ROOT)),
                "--mode",
                "full",
            ]

        else:
            st.error(f"Unknown section mode: {section_mode}")
            st.stop()

        run_cmd(cmd)
        st.success("Created sections.")

    sections = load_csv_if_exists(SECTIONS_CSV)

    if sections is not None:
        st.subheader("Detected sections")
        st.caption("You can manually adjust section labels/timestamps if the automatic split is imperfect.")

        edited_sections = st.data_editor(
            sections,
            num_rows="dynamic",
            use_container_width=True,
        )

        if st.button("Save edited sections"):
            edited_sections.to_csv(SECTIONS_CSV, index=False)
            st.success("Saved edited sections.")

    st.divider()

    st.header("4. Extract musical features")

    st.write(
        "This estimates each section's key/mode, entry/exit harmony, energy, chroma, and timbre features."
    )

    if st.button("Extract features"):
        if not TRACKS_CSV.exists() or not SECTIONS_CSV.exists():
            st.error("Need indexed tracks and sections first.")
            st.stop()

        run_cmd([
            sys.executable,
            "src/extract_section_features.py",
            "--tracks_csv",
            str(TRACKS_CSV.relative_to(PROJECT_ROOT)),
            "--sections_csv",
            str(SECTIONS_CSV.relative_to(PROJECT_ROOT)),
            "--section_features_csv",
            str(SECTION_FEATURES_CSV.relative_to(PROJECT_ROOT)),
            "--boundary_features_csv",
            str(BOUNDARY_FEATURES_CSV.relative_to(PROJECT_ROOT)),
        ])

        st.success("Extracted features.")

    section_features = load_csv_if_exists(SECTION_FEATURES_CSV)

    if section_features is not None:
        st.subheader("Extracted features")

        visible_cols = [
            c for c in [
                "section_id",
                "track_id",
                "section_label",
                "start_time",
                "end_time",
                "tempo",
                "key",
                "mode",
                "key_confidence",
                "chroma_entropy",
                "rms_mean",
            ]
            if c in section_features.columns
        ]

        st.dataframe(section_features[visible_cols], use_container_width=True)

    st.divider()

    st.header("5. Generate results")

    if planning_mode == "Full mashup routes":
        st.subheader("Full mashup routes")
        st.write(
            "Choose a mandatory starting song/section. The planner builds full routes forward from that musical state."
        )

        start_section_id = ""

        tracks_for_start = load_csv_if_exists(TRACKS_CSV)
        section_features_for_start = load_csv_if_exists(SECTION_FEATURES_CSV)

        if section_features_for_start is None:
            st.warning("Extract features first before selecting a starting section.")
        else:
            section_display = make_section_display_table(section_features_for_start, tracks_for_start)

            if len(section_display) == 0:
                st.error("No sections found. Create sections and extract features first.")
                st.stop()

            options = ["-- choose a starting song/section --"] + section_display["display_name"].tolist()

            start_display = st.selectbox(
                "Starting song/section",
                options,
                index=0,
                help=(
                    "This is required. The planner starts here, then repeatedly chooses the best next transition "
                    "from the current song/key/audio state."
                ),
            )

            if start_display != options[0]:
                matches = section_display[section_display["display_name"] == start_display]

                if not matches.empty:
                    start_section_id = matches["section_id"].iloc[0]

                    st.success("Starting section selected.")
                    st.code(start_section_id)
            else:
                st.info("Select a starting song/section before running the planner.")

        st.write("Transposition style:")
        st.code(
            f"{transposition_style} | "
            f"max shift={transposition_settings['max_abs_pitch_shift']} | "
            f"original-key bonus={transposition_settings['original_key_bonus']} | "
            f"pitch-shift penalty={transposition_settings['pitch_shift_penalty']}"
        )

        run_planner_disabled = not bool(start_section_id)

        if st.button("Run route planner", disabled=run_planner_disabled):
            if not SECTION_FEATURES_CSV.exists() or not BOUNDARY_FEATURES_CSV.exists():
                st.error("Extract features first.")
                st.stop()

            if not start_section_id:
                st.error("Choose a starting song/section first.")
                st.stop()

            model_dir = PROJECT_ROOT / "models"
            required_model_files = [
                model_dir / "reward_model.pt",
                model_dir / "reward_scaler.pkl",
                model_dir / "reward_model_config.json",
            ]

            missing = [p.name for p in required_model_files if not p.exists()]
            if missing:
                st.error(f"Missing model files: {missing}")
                st.stop()

            cmd = [
                sys.executable,
                "src/plan_mashup_with_bridges.py",
                "--section_features_csv",
                str(SECTION_FEATURES_CSV.relative_to(PROJECT_ROOT)),
                "--boundary_features_csv",
                str(BOUNDARY_FEATURES_CSV.relative_to(PROJECT_ROOT)),
                "--model_dir",
                "models",
                "--output_csv",
                str(PLANS_CSV.relative_to(PROJECT_ROOT)),
                "--output_json",
                str(PLANS_JSON.relative_to(PROJECT_ROOT)),
                "--path_length",
                str(path_length),
                "--beam_size",
                str(beam_size),
                "--initial_sections",
                str(initial_sections),
                "--candidate_edges_per_step",
                str(candidate_edges_per_step),
                "--max_abs_pitch_shift",
                str(transposition_settings["max_abs_pitch_shift"]),
                "--original_key_bonus",
                str(transposition_settings["original_key_bonus"]),
                "--pitch_shift_penalty",
                str(transposition_settings["pitch_shift_penalty"]),
                "--start_section_id",
                start_section_id,
            ]

            run_cmd(cmd)
            st.success("Generated mashup routes.")

        if run_planner_disabled:
            st.caption("Route planner is disabled until a starting song/section is selected.")

    else:
        st.subheader("Best pairwise transitions")
        st.write(
            "Rank the strongest individual handoffs anywhere in the uploaded corpus. "
            "No starting song is needed."
        )

        discovery_goal_label = st.radio(
            "Discovery goal",
            ["Smooth / compatible", "Creative / dissonant"],
            index=0,
            horizontal=True,
            help=(
                "Smooth finds seamless, compatible handoffs. "
                "Creative surfaces more surprising key relationships that still score well."
            ),
        )

        discovery_goal_arg = "creative" if discovery_goal_label == "Creative / dissonant" else "smooth"
        discovery_settings = get_discovery_settings(transposition_style, discovery_goal_arg)

        st.write("Transition ranking objective:")
        st.code(
            f"{discovery_goal_label} | {transposition_style} transposition | "
            f"showing top {pairwise_top_n} transitions | "
            f"max shift={transposition_settings['max_abs_pitch_shift']} | "
            f"original-key bonus={transposition_settings['original_key_bonus']} | "
            f"pitch-shift penalty={transposition_settings['pitch_shift_penalty']} | "
            f"discovery weight={discovery_settings['discovery_weight']}"
        )

        allow_same_track = st.checkbox(
            "Allow same-song transitions",
            value=False,
            help=(
                "By default, pairwise discovery avoids transitions between sections of the same original song. "
                "Enable this if you want to find strong internal handoffs or reprises."
            ),
        )

        section_features_for_pairwise = load_csv_if_exists(SECTION_FEATURES_CSV)

        default_exclude_final_sources = False
        if section_features_for_pairwise is not None and "track_id" in section_features_for_pairwise.columns:
            sections_per_track = section_features_for_pairwise.groupby("track_id").size()
            default_exclude_final_sources = bool(sections_per_track.max() > 1)

        exclude_final_source_sections = st.checkbox(
            "A cappella / split-song mode: avoid final song endings as transition sources",
            value=default_exclude_final_sources,
            help=(
                "When songs are split into multiple sections, this prevents final/outro sections "
                "from being used as the source of a transition. Leave this off for DJ/full-song mode."
            ),
        )

        output_csv = PAIRWISE_CREATIVE_CSV if discovery_goal_arg == "creative" else PAIRWISE_SMOOTH_CSV

        if st.button("Rank pairwise transitions"):
            if not SECTION_FEATURES_CSV.exists() or not BOUNDARY_FEATURES_CSV.exists():
                st.error("Extract features first.")
                st.stop()

            model_dir = PROJECT_ROOT / "models"
            required_model_files = [
                model_dir / "reward_model.pt",
                model_dir / "reward_scaler.pkl",
                model_dir / "reward_model_config.json",
            ]

            missing = [p.name for p in required_model_files if not p.exists()]
            if missing:
                st.error(f"Missing model files: {missing}")
                st.stop()

            cmd = [
                sys.executable,
                "src/rank_pairwise_transitions.py",
                "--section_features_csv",
                str(SECTION_FEATURES_CSV.relative_to(PROJECT_ROOT)),
                "--boundary_features_csv",
                str(BOUNDARY_FEATURES_CSV.relative_to(PROJECT_ROOT)),
                "--tracks_csv",
                str(TRACKS_CSV.relative_to(PROJECT_ROOT)),
                "--model_dir",
                "models",
                "--output_csv",
                str(output_csv.relative_to(PROJECT_ROOT)),
                "--discovery_goal",
                discovery_goal_arg,
                "--top_n",
                str(pairwise_top_n),
                "--max_candidates_to_score",
                str(max_candidates_to_score),
                "--max_abs_pitch_shift",
                str(transposition_settings["max_abs_pitch_shift"]),
                "--original_key_bonus",
                str(transposition_settings["original_key_bonus"]),
                "--pitch_shift_penalty",
                str(transposition_settings["pitch_shift_penalty"]),
                "--reward_weight",
                str(discovery_settings["reward_weight"]),
                "--theory_weight",
                str(discovery_settings["theory_weight"]),
                "--baseline_weight",
                str(discovery_settings["baseline_weight"]),
                "--discovery_weight",
                str(discovery_settings["discovery_weight"]),
            ]

            if allow_same_track:
                cmd.append("--allow_same_track")

            if exclude_final_source_sections:
                cmd.append("--exclude_final_source_sections")

            run_cmd(cmd)
            st.success("Ranked pairwise transitions.")

    if planning_mode == "Full mashup routes":
        plans = load_csv_if_exists(PLANS_CSV)

        if plans is not None:
            st.subheader("Mashup route output")

            plan_ids = sorted(plans["plan_id"].dropna().unique().tolist())
            chosen_plan = st.selectbox("Choose route", plan_ids)

            plan_rows = plans[plans["plan_id"] == chosen_plan].sort_values("step")

            tracks_for_display = load_csv_if_exists(TRACKS_CSV)

            if tracks_for_display is not None and "track_id" in tracks_for_display.columns:
                track_cols = [
                    c for c in ["track_id", "title", "artist", "path"]
                    if c in tracks_for_display.columns
                ]

                plan_rows_display = plan_rows.merge(
                    tracks_for_display[track_cols],
                    on="track_id",
                    how="left",
                )
            else:
                plan_rows_display = plan_rows.copy()

            display_cols = [
                c for c in [
                    "step",
                    "title",
                    "artist",
                    "path",
                    "section_id",
                    "original_key",
                    "realized_key",
                    "pitch_shift",
                    "incoming_from_key",
                    "incoming_to_key",
                    "incoming_harmonic_category",
                    "incoming_interval_name",
                    "incoming_reward_score",
                    "target_preserves_original_key",
                ]
                if c in plan_rows_display.columns
            ]

            st.dataframe(plan_rows_display[display_cols], use_container_width=True)

            st.subheader("Transition explanations")

            for _, row in plan_rows_display.sort_values("step").iterrows():
                step = int(row["step"])

                title = safe_display_value(row.get("title", ""), fallback="")
                artist = safe_display_value(row.get("artist", ""), fallback="")
                section_id = safe_display_value(row.get("section_id", ""), fallback="unknown_section")
                path = safe_display_value(row.get("path", ""), fallback="")

                if not title:
                    title = Path(path).stem if path else section_id

                title_text = title
                if artist:
                    title_text = f"{title_text} — {artist}"

                if step == 0:
                    with st.expander(f"Start: {title_text}", expanded=False):
                        st.write(f"**Section:** `{section_id}`")
                        if path:
                            st.write(f"**File:** `{path}`")
                        st.write("This is the starting musical state for the route.")
                    continue

                category = row.get("incoming_harmonic_category", "unknown")
                interval = row.get("incoming_interval_name", "unknown")
                from_key = row.get("incoming_from_key", "unknown")
                to_key = row.get("incoming_to_key", "unknown")
                pitch_shift = row.get("pitch_shift", 0)
                reward = row.get("incoming_reward_score", None)

                pitch_text = make_pitch_shift_text(pitch_shift)

                with st.expander(
                    f"Step {step}: {from_key} → {to_key} | {title_text}",
                    expanded=False,
                ):
                    st.write(f"**Section:** `{section_id}`")
                    st.write(f"**Harmonic move:** {from_key} → {to_key}")
                    st.write(f"**Interval:** {interval}")
                    st.write(f"**Category:** {category}")
                    st.write(f"**Pitch treatment:** {pitch_text}")

                    if reward is not None and not pd.isna(reward):
                        if str(reward).strip() != "":
                            st.write(f"**Learned reward score:** {float(reward):.3f}")

                    st.write(make_transition_explanation(row))

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.write("Pitch shifts")
                if "pitch_shift" in plans.columns:
                    st.dataframe(
                        plans["pitch_shift"]
                        .value_counts(dropna=False)
                        .rename_axis("pitch_shift")
                        .reset_index(name="count"),
                        use_container_width=True,
                    )

            with col_b:
                st.write("Harmonic categories")
                if "incoming_harmonic_category" in plans.columns:
                    st.dataframe(
                        plans["incoming_harmonic_category"]
                        .value_counts(dropna=False)
                        .rename_axis("category")
                        .reset_index(name="count"),
                        use_container_width=True,
                    )

            with col_c:
                st.write("Intervals")
                if "incoming_interval_name" in plans.columns:
                    st.dataframe(
                        plans["incoming_interval_name"]
                        .value_counts(dropna=False)
                        .rename_axis("interval")
                        .reset_index(name="count"),
                        use_container_width=True,
                    )

            st.divider()

            st.header("Render route audio previews")

            preview_mode = st.radio(
                "Preview mode",
                ["Transition previews", "Full route preview"],
                index=0,
                horizontal=True,
                help=(
                    "Transition previews are better for evaluating handoffs. "
                    "Full route preview creates one continuous rough mashup sample."
                ),
            )

            crossfade_sec = st.slider(
                "Crossfade seconds",
                0.5,
                5.0,
                2.0,
                step=0.5,
            )

            if preview_mode == "Transition previews":
                transition_context_sec = st.slider(
                    "Seconds before/after transition",
                    4,
                    20,
                    8,
                    help=(
                        "Each preview uses the last N seconds of the source section "
                        "and the first N seconds of the target section."
                    ),
                )

                if st.button("Render transition previews"):
                    OUTPUT_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

                    run_cmd([
                        sys.executable,
                        "src/render_plan_previews.py",
                        "--plans_csv",
                        str(PLANS_CSV.relative_to(PROJECT_ROOT)),
                        "--tracks_csv",
                        str(TRACKS_CSV.relative_to(PROJECT_ROOT)),
                        "--sections_csv",
                        str(SECTION_FEATURES_CSV.relative_to(PROJECT_ROOT)),
                        "--output_dir",
                        str(OUTPUT_PREVIEW_DIR.relative_to(PROJECT_ROOT)),
                        "--plan_id",
                        chosen_plan,
                        "--mode",
                        "transitions",
                        "--crossfade_sec",
                        str(crossfade_sec),
                        "--transition_context_sec",
                        str(transition_context_sec),
                    ])

                    st.success("Rendered transition previews.")

                transition_paths = sorted(
                    OUTPUT_PREVIEW_DIR.glob(f"{chosen_plan}_transition_*.wav")
                )

                if transition_paths:
                    st.subheader("Transition preview audio")

                    for path in transition_paths:
                        st.write(path.name)
                        st.audio(str(path))

            else:
                preview_section_duration = st.slider(
                    "Seconds per section in full-route preview",
                    8,
                    30,
                    12,
                    help="Uses the first N seconds of each chosen section to create a compact route preview.",
                )

                if st.button("Render full route preview"):
                    OUTPUT_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

                    run_cmd([
                        sys.executable,
                        "src/render_plan_previews.py",
                        "--plans_csv",
                        str(PLANS_CSV.relative_to(PROJECT_ROOT)),
                        "--tracks_csv",
                        str(TRACKS_CSV.relative_to(PROJECT_ROOT)),
                        "--sections_csv",
                        str(SECTION_FEATURES_CSV.relative_to(PROJECT_ROOT)),
                        "--output_dir",
                        str(OUTPUT_PREVIEW_DIR.relative_to(PROJECT_ROOT)),
                        "--plan_id",
                        chosen_plan,
                        "--mode",
                        "full_plan",
                        "--crossfade_sec",
                        str(crossfade_sec),
                        "--section_duration_sec",
                        str(preview_section_duration),
                    ])

                    st.success("Rendered full route preview.")

                preview_path = OUTPUT_PREVIEW_DIR / f"{chosen_plan}_preview.wav"

                if preview_path.exists():
                    st.audio(str(preview_path))
                    st.write(preview_path.relative_to(PROJECT_ROOT))

    else:
        pairwise_csv = PAIRWISE_CREATIVE_CSV if discovery_goal_arg == "creative" else PAIRWISE_SMOOTH_CSV
        pairwise = load_csv_if_exists(pairwise_csv)

        if pairwise is not None:
            st.subheader("Ranked pairwise transitions")

            tracks_for_display = load_csv_if_exists(TRACKS_CSV)

            pairwise_display = merge_tracks_for_display(
                pairwise,
                tracks_for_display,
                source_col="source_track_id",
                target_col="target_track_id",
            )

            display_cols = [
                c for c in [
                    "rank",
                    "source_title",
                    "target_title",
                    "source_section_id",
                    "target_section_id",
                    "source_exit_key",
                    "source_exit_mode",
                    "target_original_key",
                    "target_realized_key",
                    "target_pitch_shift",
                    "interval_name",
                    "harmonic_category",
                    "reward_score",
                    "baseline_score",
                    "discovery_bonus",
                    "final_edge_score",
                ]
                if c in pairwise_display.columns
            ]

            st.dataframe(pairwise_display[display_cols], use_container_width=True)

            st.write("Harmonic categories in ranked results:")
            st.dataframe(
                pairwise["harmonic_category"]
                .value_counts(dropna=False)
                .rename_axis("category")
                .reset_index(name="count"),
                use_container_width=True,
            )

            st.subheader("Inspect and preview one transition")

            choice_options = []
            for _, row in pairwise_display.iterrows():
                source_title = safe_display_value(row.get("source_title", ""), fallback=row["source_section_id"])
                target_title = safe_display_value(row.get("target_title", ""), fallback=row["target_section_id"])
                rank = int(row["rank"])
                category = row.get("harmonic_category", "unknown")
                interval = row.get("interval_name", "unknown")
                shift = row.get("target_pitch_shift", 0)
                choice_options.append(
                    f"#{rank}: {source_title} → {target_title} | {category} / {interval} | shift {shift}"
                )

            selected_transition = st.selectbox(
                "Choose a transition to inspect",
                choice_options,
                index=0,
            )

            selected_idx = choice_options.index(selected_transition)
            selected_row = pairwise_display.iloc[selected_idx]

            with st.expander("Transition explanation", expanded=True):
                st.write(
                    f"**Move:** {selected_row.get('source_exit_key', '?')} {selected_row.get('source_exit_mode', '')} "
                    f"→ {selected_row.get('target_realized_key', '?')} {selected_row.get('target_entry_mode', '')}"
                )
                st.write(f"**Interval:** {selected_row.get('interval_name', 'unknown')}")
                st.write(f"**Category:** {selected_row.get('harmonic_category', 'unknown')}")
                st.write(f"**Pitch treatment:** {make_pitch_shift_text(selected_row.get('target_pitch_shift', 0))}")
                st.write(f"**Final edge score:** {float(selected_row.get('final_edge_score', 0)):.3f}")
                st.write(make_transition_explanation(selected_row))

            transition_context_sec = st.slider(
                "Seconds before/after selected transition",
                4,
                20,
                8,
            )

            crossfade_sec_pairwise = st.slider(
                "Selected transition crossfade seconds",
                0.5,
                5.0,
                2.0,
                step=0.5,
            )

            if st.button("Render selected pairwise transition"):
                OUTPUT_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

                temp_plan_csv = UPLOAD_PROCESSED_DIR / "selected_pairwise_transition_plan.csv"

                write_pairwise_transition_as_plan_csv(
                    selected_row,
                    temp_plan_csv,
                )

                run_cmd([
                    sys.executable,
                    "src/render_plan_previews.py",
                    "--plans_csv",
                    str(temp_plan_csv.relative_to(PROJECT_ROOT)),
                    "--tracks_csv",
                    str(TRACKS_CSV.relative_to(PROJECT_ROOT)),
                    "--sections_csv",
                    str(SECTION_FEATURES_CSV.relative_to(PROJECT_ROOT)),
                    "--output_dir",
                    str(OUTPUT_PREVIEW_DIR.relative_to(PROJECT_ROOT)),
                    "--plan_id",
                    "PAIRWISE_SELECTED",
                    "--mode",
                    "transitions",
                    "--crossfade_sec",
                    str(crossfade_sec_pairwise),
                    "--transition_context_sec",
                    str(transition_context_sec),
                ])

                st.success("Rendered selected pairwise transition.")

            selected_preview = OUTPUT_PREVIEW_DIR / "PAIRWISE_SELECTED_transition_01.wav"

            if selected_preview.exists():
                st.audio(str(selected_preview))
                st.write(selected_preview.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()