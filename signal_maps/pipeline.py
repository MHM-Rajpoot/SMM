from __future__ import annotations

from pathlib import Path
import math
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .io_utils import ArrayLike, create_run_dir, load_signal, window_indices
from .time_frequency import compute_cwt_map, compute_stft_map, generate_time_frequency_maps
from .topomap import (
    channel_layout,
    generate_topographic_maps,
    interpolate_topomap,
    virtualize_to_19_channels,
)
from .types import CombinedOutputPaths, TimeFrequencyOutputPaths, TopomapOutputPaths


def _to_single_channel(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[..., :3].mean(axis=2)
    raise ValueError("Unsupported image shape for map concatenation.")


def _read_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im)


def _resize_nearest(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    if src_h == target_h and src_w == target_w:
        return img
    y_idx = np.linspace(0, src_h - 1, target_h).astype(np.int64)
    x_idx = np.linspace(0, src_w - 1, target_w).astype(np.int64)
    return img[y_idx][:, x_idx]


def _concat_two_arrays(tf_img: np.ndarray, topo_img: np.ndarray) -> np.ndarray:
    tf_img = _to_single_channel(tf_img)
    topo_img = _to_single_channel(topo_img)
    target_h = min(tf_img.shape[0], topo_img.shape[0])
    target_w = min(tf_img.shape[1], topo_img.shape[1])
    tf_rs = _resize_nearest(tf_img, target_h, target_w)
    topo_rs = _resize_nearest(topo_img, target_h, target_w)
    return np.stack([tf_rs, topo_rs], axis=-1)


def _concat_two_maps(tf_path: Path, topo_path: Path) -> np.ndarray:
    tf_img = _read_image(tf_path)
    topo_img = _read_image(topo_path)
    return _concat_two_arrays(tf_img, topo_img)


def _save_combined_preview(combined_xy2: np.ndarray, out_file: Path) -> None:
    ch0 = combined_xy2[..., 0]
    ch1 = combined_xy2[..., 1]

    def norm01(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        valid = np.isfinite(x)
        if not np.any(valid):
            return np.zeros_like(x, dtype=np.float64)
        lo = float(np.percentile(x[valid], 1))
        hi = float(np.percentile(x[valid], 99))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float64)
        y = np.zeros_like(x, dtype=np.float64)
        y[valid] = (x[valid] - lo) / (hi - lo)
        y = np.clip(y, 0.0, 1.0)
        y[~valid] = 0.0
        return y

    ch0n = norm01(ch0)
    ch1n = norm01(ch1)
    side_by_side = np.concatenate([ch0n, ch1n], axis=1)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=140)
    im = ax.imshow(side_by_side, cmap="magma", origin="upper", aspect="auto")
    ax.set_title("Combined Preview (left=Time-Frequency, right=Topomap)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="Normalized amplitude")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def _save_combined_frames_preview(
    combined_frames_nxy2: np.ndarray,
    out_file: Path,
    max_frames: int = 8,
    gap: int = 4,
) -> None:
    arr = np.asarray(combined_frames_nxy2, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[-1] != 2 or arr.shape[0] == 0:
        empty = np.zeros((64, 128, 3), dtype=np.uint8)
        Image.fromarray(empty, mode="RGB").save(out_file)
        return

    n = arr.shape[0]
    count = min(max_frames, n)
    idx = np.linspace(0, n - 1, num=count).astype(np.int64)
    idx = np.unique(idx)

    tiles = []
    for i in idx:
        frame = arr[i]
        ch0 = frame[..., 0]
        ch1 = frame[..., 1]

        def norm01(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            valid = np.isfinite(x)
            if not np.any(valid):
                return np.zeros_like(x, dtype=np.float32)
            lo = float(np.percentile(x[valid], 1))
            hi = float(np.percentile(x[valid], 99))
            if hi <= lo:
                return np.zeros_like(x, dtype=np.float32)
            y = np.zeros_like(x, dtype=np.float32)
            y[valid] = (x[valid] - lo) / (hi - lo)
            y = np.clip(y, 0.0, 1.0)
            y[~valid] = 0.0
            return y

        side = np.concatenate([norm01(ch0), norm01(ch1)], axis=1)
        tile = (side * 255.0).astype(np.uint8)
        tiles.append(np.repeat(tile[..., np.newaxis], 3, axis=2))

    if not tiles:
        empty = np.zeros((64, 128, 3), dtype=np.uint8)
        Image.fromarray(empty, mode="RGB").save(out_file)
        return

    tile_h, tile_w, _ = tiles[0].shape
    cols = min(4, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    canvas_h = rows * tile_h + (rows - 1) * gap
    canvas_w = cols * tile_w + (cols - 1) * gap
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for k, tile in enumerate(tiles):
        r = k // cols
        c = k % cols
        y = r * (tile_h + gap)
        x = c * (tile_w + gap)
        canvas[y : y + tile_h, x : x + tile_w] = tile

    Image.fromarray(canvas, mode="RGB").save(out_file)


def _write_shapes_file(combined_single: np.ndarray, combined_frames: np.ndarray, root: Path) -> Path:
    out = root / "combined_shapes.txt"
    lines = [
        f"combined_single_xy2.npy: {tuple(combined_single.shape)}",
        f"combined_frames_nxy2.npy: {tuple(combined_frames.shape)}",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _save_combined_channel_npz(combined_frames_nxy2: np.ndarray, root: Path) -> None:
    arr = np.asarray(combined_frames_nxy2)
    if arr.ndim != 4 or arr.shape[-1] != 2:
        freq = np.zeros((0, 0, 0), dtype=np.float32)
        topo = np.zeros((0, 0, 0), dtype=np.float32)
    else:
        freq = arr[..., 0]
        topo = arr[..., 1]
    np.savez_compressed(root / "combined_freq.npz", data=freq)
    np.savez_compressed(root / "combined_topo.npz", data=topo)


def _normalize_output_mode(output_mode: str) -> str:
    mode = output_mode.lower().strip()
    if mode in {"all", "medium", "least"}:
        return mode
    raise ValueError(f"Unsupported output_mode: {output_mode}")


def _tf_amp(x: np.ndarray, fs: int, method: str) -> np.ndarray:
    if method.lower() == "stft":
        _, _, amp = compute_stft_map(x, fs)
        return amp
    _, _, amp = compute_cwt_map(x, fs)
    return amp


def _build_compact_combined(
    raw: np.ndarray,
    fs: int,
    root: Path,
    tf_method: str,
    window_sec: float,
    step_sec: float,
    channel_names: Optional[Sequence[str]],
    save_preview: bool,
) -> CombinedOutputPaths:
    mono = np.mean(raw, axis=0)

    if raw.shape[0] < 4:
        topo_src = virtualize_to_19_channels(mono, fs)
        topo_names = None
    else:
        topo_src = raw
        topo_names = channel_names

    _, xy = channel_layout(topo_names, topo_src.shape[0])

    all_frames = []
    tf_windows = list(window_indices(mono.shape[0], fs, window_sec, step_sec))
    topo_windows = list(window_indices(topo_src.shape[1], fs, window_sec, step_sec))
    pair_count = min(len(tf_windows), len(topo_windows))

    for idx in range(pair_count):
        _, t_start, t_end = tf_windows[idx]
        _, p_start, p_end = topo_windows[idx]

        tf_seg = mono[t_start:t_end]
        tf_amp = _tf_amp(tf_seg, fs, tf_method)

        topo_amp = np.sqrt(np.mean(topo_src[:, p_start:p_end] ** 2, axis=1))
        topo_grid = interpolate_topomap(topo_amp, xy)

        all_frames.append(_concat_two_arrays(tf_amp, topo_grid))

    combined_frames_file = root / "combined_frames_nxy2.npy"
    combined_frames_data: np.ndarray
    if all_frames:
        combined_frames_data = np.stack(all_frames, axis=0)
    else:
        combined_frames_data = np.zeros((0, 0, 0, 2), dtype=np.float32)
    np.save(combined_frames_file, combined_frames_data)
    _save_combined_channel_npz(combined_frames_data, root)
    preview_file = root / "combined_preview.png"
    if save_preview:
        single_tf = _tf_amp(mono, fs, tf_method)
        single_topo = interpolate_topomap(np.sqrt(np.mean(topo_src**2, axis=1)), xy)
        combined_single = _concat_two_arrays(single_tf, single_topo)
        _save_combined_preview(combined_single, preview_file)

    tf_out = TimeFrequencyOutputPaths(
        root=root,
        single_file=root / "time_frequency_single.png",
        frames_dir=root / "time_frequency_frames",
    )
    topo_out = TopomapOutputPaths(
        root=root,
        single_file=root / "topomap_single.png",
        frames_dir=root / "topomap_frames",
    )

    return CombinedOutputPaths(
        root=root,
        time_frequency=tf_out,
        topomap=topo_out,
        combined_single_file=root / "combined_single_xy2.npy",
        combined_frames_file=combined_frames_file,
        combined_frames_dir=root / "combined_frames_xy2",
    )


def generate_both_maps(
    signal_data: Optional[ArrayLike] = None,
    sample_rate: Optional[int] = None,
    input_file: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "outputs",
    tf_method: str = "cwt",
    window_sec: float = 2.0,
    step_sec: float = 1.0,
    channel_names: Optional[Sequence[str]] = None,
    output_mode: str = "all",
) -> CombinedOutputPaths:
    raw, fs = load_signal(signal_data=signal_data, sample_rate=sample_rate, input_file=input_file)
    root = create_run_dir(output_dir, use_timestamp=False, overwrite=True)
    mode = _normalize_output_mode(output_mode)

    if mode in {"medium", "least"}:
        return _build_compact_combined(
            raw=raw,
            fs=fs,
            root=root,
            tf_method=tf_method,
            window_sec=window_sec,
            step_sec=step_sec,
            channel_names=channel_names,
            save_preview=(mode == "medium"),
        )

    tf_out = generate_time_frequency_maps(
        signal_data=raw,
        sample_rate=fs,
        output_dir=output_dir,
        method=tf_method,
        window_sec=window_sec,
        step_sec=step_sec,
        root_dir=root,
    )
    topo_out = generate_topographic_maps(
        signal_data=raw,
        sample_rate=fs,
        output_dir=output_dir,
        window_sec=window_sec,
        step_sec=step_sec,
        channel_names=channel_names,
        root_dir=root,
    )

    combined_single = _concat_two_maps(tf_out.single_file, topo_out.single_file)
    combined_single_file = root / "combined_single_xy2.npy"
    np.save(combined_single_file, combined_single)

    combined_frames_dir = root / "combined_frames_xy2"
    combined_frames_dir.mkdir(parents=True, exist_ok=True)
    tf_frames = sorted(tf_out.frames_dir.glob("*.png"))
    topo_frames = sorted(topo_out.frames_dir.glob("*.png"))
    pair_count = min(len(tf_frames), len(topo_frames))
    all_frames = []
    for idx in range(pair_count):
        frame_xy2 = _concat_two_maps(tf_frames[idx], topo_frames[idx])
        np.save(combined_frames_dir / f"combined_{idx:04d}.npy", frame_xy2)
        all_frames.append(frame_xy2)

    combined_frames_file = root / "combined_frames_nxy2.npy"
    combined_frames_data: np.ndarray
    if all_frames:
        combined_frames_data = np.stack(all_frames, axis=0)
    else:
        combined_frames_data = np.zeros((0, 0, 0, 2), dtype=np.float32)
    np.save(combined_frames_file, combined_frames_data)
    _save_combined_channel_npz(combined_frames_data, root)
    _write_shapes_file(combined_single=combined_single, combined_frames=combined_frames_data, root=root)

    preview_file = root / "combined_preview.png"
    _save_combined_preview(combined_single, preview_file)
    frames_preview_file = root / "combined_frames_preview.png"
    _save_combined_frames_preview(combined_frames_data, frames_preview_file)

    return CombinedOutputPaths(
        root=root,
        time_frequency=tf_out,
        topomap=topo_out,
        combined_single_file=combined_single_file,
        combined_frames_file=combined_frames_file,
        combined_frames_dir=combined_frames_dir,
    )
