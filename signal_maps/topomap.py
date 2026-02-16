from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps
import numpy as np
from PIL import Image
from scipy import signal
from scipy.interpolate import griddata

from .io_utils import ArrayLike, create_run_dir, load_signal, window_indices
from .types import TopomapOutputPaths

STANDARD_1020_POSITIONS = {
    "Fp1": (-0.5, 1.0),
    "Fp2": (0.5, 1.0),
    "F7": (-1.0, 0.5),
    "F3": (-0.5, 0.5),
    "Fz": (0.0, 0.5),
    "F4": (0.5, 0.5),
    "F8": (1.0, 0.5),
    "T7": (-1.1, 0.0),
    "C3": (-0.5, 0.0),
    "Cz": (0.0, 0.0),
    "C4": (0.5, 0.0),
    "T8": (1.1, 0.0),
    "P7": (-1.0, -0.5),
    "P3": (-0.5, -0.5),
    "Pz": (0.0, -0.5),
    "P4": (0.5, -0.5),
    "P8": (1.0, -0.5),
    "O1": (-0.5, -1.0),
    "O2": (0.5, -1.0),
}


def virtualize_to_19_channels(x: np.ndarray, fs: int) -> np.ndarray:
    target = 19
    nyquist = fs / 2.0
    high = min(45.0, nyquist - 1.0)

    if high <= 2.5:
        return np.vstack([np.roll(x, i * 5) for i in range(target)])

    centers = np.linspace(1.5, high, target)
    channels = []
    for c in centers:
        low = max(0.5, c - 1.0)
        hi = min(high, c + 1.0)
        if hi <= low:
            channels.append(np.roll(x, len(channels) * 5))
            continue
        try:
            sos = signal.butter(3, [low, hi], btype="bandpass", fs=fs, output="sos")
            filt = signal.sosfiltfilt(sos, x)
            channels.append(filt)
        except Exception:
            channels.append(np.roll(x, len(channels) * 5))
    return np.vstack(channels)


def channel_layout(channel_names: Optional[Sequence[str]], n_channels: int):
    all_names = list(STANDARD_1020_POSITIONS.keys())
    if channel_names is None:
        names = all_names[:n_channels]
    else:
        names = [n for n in channel_names if n in STANDARD_1020_POSITIONS]
        if len(names) < n_channels:
            names += [n for n in all_names if n not in names]
            names = names[:n_channels]

    xy = np.array([STANDARD_1020_POSITIONS[n] for n in names], dtype=np.float64)
    return names, xy


def interpolate_topomap(values: np.ndarray, xy: np.ndarray, grid_size: int = 160):
    gx = np.linspace(-1.2, 1.2, grid_size)
    gy = np.linspace(-1.2, 1.2, grid_size)
    grid_x, grid_y = np.meshgrid(gx, gy)

    grid = griddata(xy, values, (grid_x, grid_y), method="cubic")
    if np.any(np.isnan(grid)):
        nearest = griddata(xy, values, (grid_x, grid_y), method="nearest")
        grid = np.where(np.isnan(grid), nearest, grid)

    mask = (grid_x**2 + grid_y**2) > 1.2**2
    grid = np.where(mask, np.nan, grid)
    return grid


def plot_topomap(grid: np.ndarray, out_file: Path, title: str) -> None:
    _ = title
    valid = np.isfinite(grid)
    if not np.any(valid):
        rgb = np.zeros((*grid.shape, 3), dtype=np.uint8)
        Image.fromarray(rgb, mode="RGB").save(out_file)
        return

    lo = float(np.percentile(grid[valid], 1))
    hi = float(np.percentile(grid[valid], 99))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.zeros_like(grid, dtype=np.float64)
    norm[valid] = np.clip((grid[valid] - lo) / (hi - lo), 0.0, 1.0)
    rgb = (colormaps["viridis"](norm)[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    Image.fromarray(rgb, mode="RGB").save(out_file)


def generate_topographic_maps(
    signal_data: Optional[ArrayLike] = None,
    sample_rate: Optional[int] = None,
    input_file: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "outputs",
    window_sec: float = 2.0,
    step_sec: float = 1.0,
    channel_names: Optional[Sequence[str]] = None,
    root_dir: Optional[Union[str, Path]] = None,
) -> TopomapOutputPaths:
    raw, fs = load_signal(signal_data=signal_data, sample_rate=sample_rate, input_file=input_file)
    root = Path(root_dir) if root_dir is not None else create_run_dir(output_dir)

    frames_dir = root / "topomap_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    single_file = root / "topomap_single.png"

    mono = np.mean(raw, axis=0)
    if raw.shape[0] < 4:
        topo_src = virtualize_to_19_channels(mono, fs)
        topo_names = list(STANDARD_1020_POSITIONS.keys())
    else:
        topo_src = raw
        topo_names = channel_names

    _, xy = channel_layout(topo_names, topo_src.shape[0])

    channel_amp = np.sqrt(np.mean(topo_src**2, axis=1))
    single_grid = interpolate_topomap(channel_amp, xy)
    plot_topomap(single_grid, single_file, "Topographic Amplitude Map")

    for idx, start, end in window_indices(topo_src.shape[1], fs, window_sec, step_sec):
        w_amp = np.sqrt(np.mean(topo_src[:, start:end] ** 2, axis=1))
        grid = interpolate_topomap(w_amp, xy)
        title = f"Topomap Window {idx} ({start/fs:.2f}s-{end/fs:.2f}s)"
        plot_topomap(grid, frames_dir / f"topomap_{idx:04d}.png", title)

    return TopomapOutputPaths(root=root, single_file=single_file, frames_dir=frames_dir)
