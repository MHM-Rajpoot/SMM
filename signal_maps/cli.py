from __future__ import annotations

import argparse

from .pipeline import generate_both_maps
from .time_frequency import generate_time_frequency_maps
from .topomap import generate_topographic_maps


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate time-frequency maps and EEG topographic maps.")
    parser.add_argument("--input", type=str, required=True, help="Input .wav or .mp3 file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument("--window-sec", type=float, default=2.0, help="Window size in seconds")
    parser.add_argument("--step-sec", type=float, default=1.0, help="Step size in seconds")
    parser.add_argument("--tf-method", choices=["cwt", "stft"], default="cwt", help="Time-frequency method")
    parser.add_argument(
        "--output-mode",
        choices=["all", "medium", "least"],
        default="all",
        help="all: save every output; medium: save combined_frames_nxy2.npy + combined_preview.png; least: save only combined_frames_nxy2.npy",
    )
    parser.add_argument(
        "--task",
        choices=["both", "tf", "topomap"],
        default="both",
        help="What to generate: both maps, time-frequency only, or topomap only",
    )
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    if args.task == "tf":
        out = generate_time_frequency_maps(
            input_file=args.input,
            output_dir=args.output_dir,
            method=args.tf_method,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
        )
        print(f"Created output folder: {out.root}")
        print(f"Single time-frequency map: {out.single_file}")
        print(f"Time-frequency frame folder: {out.frames_dir}")
        return

    if args.task == "topomap":
        out = generate_topographic_maps(
            input_file=args.input,
            output_dir=args.output_dir,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
        )
        print(f"Created output folder: {out.root}")
        print(f"Single topomap: {out.single_file}")
        print(f"Topomap frame folder: {out.frames_dir}")
        return

    out = generate_both_maps(
        input_file=args.input,
        output_dir=args.output_dir,
        tf_method=args.tf_method,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        output_mode=args.output_mode,
    )
    print(f"Created output folder: {out.root}")
    mode = args.output_mode.lower()

    if mode == "least":
        print(f"Combined frames tensor (n,x,y,2): {out.combined_frames_file}")
        print(f"Combined freq npz: {out.root / 'combined_freq.npz'}")
        print(f"Combined topo npz: {out.root / 'combined_topo.npz'}")
    elif mode == "medium":
        print(f"Combined frames tensor (n,x,y,2): {out.combined_frames_file}")
        print(f"Combined freq npz: {out.root / 'combined_freq.npz'}")
        print(f"Combined topo npz: {out.root / 'combined_topo.npz'}")
        print(f"Combined preview image: {out.root / 'combined_preview.png'}")
    else:
        print(f"Single time-frequency map: {out.time_frequency.single_file}")
        print(f"Time-frequency frame folder: {out.time_frequency.frames_dir}")
        print(f"Single topomap: {out.topomap.single_file}")
        print(f"Topomap frame folder: {out.topomap.frames_dir}")
        print(f"Combined single tensor (x,y,2): {out.combined_single_file}")
        print(f"Combined frames tensor (n,x,y,2): {out.combined_frames_file}")
        print(f"Combined freq npz: {out.root / 'combined_freq.npz'}")
        print(f"Combined topo npz: {out.root / 'combined_topo.npz'}")
        print(f"Combined frames preview image: {out.root / 'combined_frames_preview.png'}")
        print(f"Combined per-frame folder: {out.combined_frames_dir}")


if __name__ == "__main__":
    main()
