from pathlib import Path
import argparse

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


SR = 22050


def normalize_audio(y, peak=0.9):
    if len(y) == 0:
        return y

    max_abs = np.max(np.abs(y))
    if max_abs == 0:
        return y

    return y / max_abs * peak


def apply_pitch_shift(y, pitch_shift, sr=SR):
    if pd.isna(pitch_shift) or str(pitch_shift).strip() == "":
        pitch_shift = 0

    pitch_shift = int(float(pitch_shift))

    if pitch_shift == 0:
        return y

    return librosa.effects.pitch_shift(
        y,
        sr=sr,
        n_steps=pitch_shift,
    )


def crossfade(a, b, sr=SR, crossfade_sec=2.0):
    fade_len = int(crossfade_sec * sr)

    if len(a) == 0:
        return b

    if len(b) == 0:
        return a

    if fade_len <= 0:
        return np.concatenate([a, b])

    if len(a) < fade_len or len(b) < fade_len:
        return np.concatenate([a, b])

    a_main = a[:-fade_len]
    a_fade = a[-fade_len:]

    b_fade = b[:fade_len]
    b_main = b[fade_len:]

    fade_out = np.linspace(1.0, 0.0, fade_len)
    fade_in = np.linspace(0.0, 1.0, fade_len)

    blend = a_fade * fade_out + b_fade * fade_in

    return np.concatenate([a_main, blend, b_main])


def load_audio_segment(path, start_time, end_time, pitch_shift=0):
    start_time = max(0.0, float(start_time))
    end_time = max(start_time + 0.1, float(end_time))
    duration = end_time - start_time

    y, sr = librosa.load(
        path,
        sr=SR,
        mono=True,
        offset=start_time,
        duration=duration,
    )

    y = apply_pitch_shift(y, pitch_shift, sr=sr)
    return normalize_audio(y, peak=0.85)


def get_section(section_id, sections_by_id):
    if section_id not in sections_by_id.index:
        raise KeyError(f"section_id not found in sections.csv: {section_id}")

    return sections_by_id.loc[section_id]


def get_track_path(track_id, tracks_by_id):
    if track_id not in tracks_by_id.index:
        raise KeyError(f"track_id not found in tracks.csv: {track_id}")

    return tracks_by_id.loc[track_id]["path"]


def render_full_plan(
    plan_rows,
    tracks_by_id,
    sections_by_id,
    output_path,
    crossfade_sec=2.0,
    section_duration_sec=12.0,
):
    rendered = np.array([], dtype=np.float32)

    for _, row in plan_rows.iterrows():
        section = get_section(row["section_id"], sections_by_id)
        path = get_track_path(section["track_id"], tracks_by_id)

        start_time = float(section["start_time"])
        end_time = float(section["end_time"])

        if section_duration_sec is not None and section_duration_sec > 0:
            end_time = min(end_time, start_time + section_duration_sec)

        pitch_shift = row.get("pitch_shift", 0)

        y = load_audio_segment(
            path=path,
            start_time=start_time,
            end_time=end_time,
            pitch_shift=pitch_shift,
        )

        if len(rendered) == 0:
            rendered = y
        else:
            rendered = crossfade(
                rendered,
                y,
                sr=SR,
                crossfade_sec=crossfade_sec,
            )

    rendered = normalize_audio(rendered, peak=0.95)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, rendered, SR)

    return output_path


def render_transition_preview(
    prev_row,
    curr_row,
    tracks_by_id,
    sections_by_id,
    output_path,
    transition_context_sec=8.0,
    crossfade_sec=2.0,
):
    """
    Render:
      last N seconds of previous/source section
      -> crossfade
      -> first N seconds of current/target section

    The target section receives the pitch_shift recommended by the planner.
    """
    source_section = get_section(prev_row["section_id"], sections_by_id)
    target_section = get_section(curr_row["section_id"], sections_by_id)

    source_path = get_track_path(source_section["track_id"], tracks_by_id)
    target_path = get_track_path(target_section["track_id"], tracks_by_id)

    source_start = float(source_section["start_time"])
    source_end = float(source_section["end_time"])

    target_start = float(target_section["start_time"])
    target_end = float(target_section["end_time"])

    # Last N seconds of source.
    source_clip_start = max(source_start, source_end - transition_context_sec)
    source_clip_end = source_end

    # First N seconds of target.
    target_clip_start = target_start
    target_clip_end = min(target_end, target_start + transition_context_sec)

    # Source is already in current realized state in the plan. For now, the row's
    # pitch shift is only applied to the target/current row.
    source_pitch_shift = prev_row.get("pitch_shift", 0)
    target_pitch_shift = curr_row.get("pitch_shift", 0)

    source_audio = load_audio_segment(
        path=source_path,
        start_time=source_clip_start,
        end_time=source_clip_end,
        pitch_shift=source_pitch_shift,
    )

    target_audio = load_audio_segment(
        path=target_path,
        start_time=target_clip_start,
        end_time=target_clip_end,
        pitch_shift=target_pitch_shift,
    )

    rendered = crossfade(
        source_audio,
        target_audio,
        sr=SR,
        crossfade_sec=crossfade_sec,
    )

    rendered = normalize_audio(rendered, peak=0.95)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, rendered, SR)

    return output_path


def render_all_transition_previews(
    plan_rows,
    tracks_by_id,
    sections_by_id,
    output_dir,
    plan_id,
    transition_context_sec=8.0,
    crossfade_sec=2.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []

    plan_rows = plan_rows.sort_values("step").reset_index(drop=True)

    for i in range(1, len(plan_rows)):
        prev_row = plan_rows.iloc[i - 1]
        curr_row = plan_rows.iloc[i]

        step = int(curr_row["step"])
        output_path = output_dir / f"{plan_id}_transition_{step:02d}.wav"

        render_transition_preview(
            prev_row=prev_row,
            curr_row=curr_row,
            tracks_by_id=tracks_by_id,
            sections_by_id=sections_by_id,
            output_path=output_path,
            transition_context_sec=transition_context_sec,
            crossfade_sec=crossfade_sec,
        )

        outputs.append(output_path)

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plans_csv",
        default="data/user_uploads/processed/mashup_plans_with_bridges.csv",
    )
    parser.add_argument(
        "--tracks_csv",
        default="data/user_uploads/processed/tracks.csv",
    )
    parser.add_argument(
        "--sections_csv",
        default="data/user_uploads/processed/sections.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/user_previews",
    )
    parser.add_argument(
        "--plan_id",
        default="PLAN000",
    )
    parser.add_argument(
        "--mode",
        choices=["full_plan", "transitions"],
        default="transitions",
    )
    parser.add_argument(
        "--crossfade_sec",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--section_duration_sec",
        type=float,
        default=12.0,
        help="For full_plan mode: use first N seconds of each section. Set <=0 for full section.",
    )
    parser.add_argument(
        "--transition_context_sec",
        type=float,
        default=8.0,
        help="For transitions mode: seconds before/after each transition.",
    )

    args = parser.parse_args()

    plans = pd.read_csv(args.plans_csv)
    tracks = pd.read_csv(args.tracks_csv)
    sections = pd.read_csv(args.sections_csv)

    tracks_by_id = tracks.set_index("track_id")
    sections_by_id = sections.set_index("section_id")

    plan_rows = plans[plans["plan_id"] == args.plan_id].copy()

    if plan_rows.empty:
        raise ValueError(f"No rows found for plan_id={args.plan_id}")

    plan_rows = plan_rows.sort_values("step")

    output_dir = Path(args.output_dir)

    print(f"Rendering {args.plan_id} in mode={args.mode}...")
    preview_cols = [
        "step",
        "section_id",
        "track_id",
        "original_key",
        "realized_key",
        "pitch_shift",
        "incoming_harmonic_category",
        "incoming_interval_name",
    ]
    existing = [c for c in preview_cols if c in plan_rows.columns]
    print(plan_rows[existing].to_string(index=False))

    if args.mode == "full_plan":
        section_duration_sec = args.section_duration_sec
        if section_duration_sec <= 0:
            section_duration_sec = None

        output_path = output_dir / f"{args.plan_id}_preview.wav"

        out = render_full_plan(
            plan_rows=plan_rows,
            tracks_by_id=tracks_by_id,
            sections_by_id=sections_by_id,
            output_path=output_path,
            crossfade_sec=args.crossfade_sec,
            section_duration_sec=section_duration_sec,
        )

        print(f"\nSaved full-plan preview to {out}")

    else:
        outputs = render_all_transition_previews(
            plan_rows=plan_rows,
            tracks_by_id=tracks_by_id,
            sections_by_id=sections_by_id,
            output_dir=output_dir,
            plan_id=args.plan_id,
            transition_context_sec=args.transition_context_sec,
            crossfade_sec=args.crossfade_sec,
        )

        print("\nSaved transition previews:")
        for out in outputs:
            print(out)


if __name__ == "__main__":
    main()