from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps
import numpy as np
from PIL import Image
from scipy import signal

from .io_utils import ArrayLike, create_run_dir, load_signal, window_indices
from .types import TimeFrequencyOutputPaths


def compute_stft_map(x: np.ndarray, fs: int, nperseg: int = 256, noverlap: int = 128):
    freqs, times, spec = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    amp = np.abs(spec)
    return freqs, times, amp


def compute_cwt_map(x: np.ndarray, fs: int, fmin: float = 1.0, fmax: float = 45.0, bins: int = 64):
    if not hasattr(signal, "cwt") or not hasattr(signal, "morlet2"):
        freqs, times, amp = compute_stft_map(x, fs)
        return freqs, times, amp

    w = 6.0
    nyquist = (fs / 2.0) - 1e-6
    fmax = max(fmin + 1e-6, min(fmax, nyquist))
    freqs = np.linspace(fmin, fmax, bins)
    widths = (w * fs) / (2.0 * np.pi * freqs)
    cwt_mat = signal.cwt(x, signal.morlet2, widths, w=w)
    amp = np.abs(cwt_mat)
    times = np.arange(x.shape[-1], dtype=np.float64) / float(fs)
    return freqs, times, amp


def plot_tf_image(freqs: np.ndarray, times: np.ndarray, amp: np.ndarray, out_file: Path, title: str) -> None:
    disp = np.log1p(np.asarray(amp, dtype=np.float64))
    if np.allclose(disp.max(), disp.min()):
        vmin, vmax = float(disp.min()), float(disp.max() + 1e-6)
    else:
        vmin = float(np.percentile(disp, 5))
        vmax = float(np.percentile(disp, 99.5))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    _ = (freqs, times, title)
    norm = np.clip((disp - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = (colormaps["magma"](norm)[..., :3] * 255.0).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(out_file)


def generate_time_frequency_maps(
    signal_data: Optional[ArrayLike] = None,
    sample_rate: Optional[int] = None,
    input_file: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "outputs",
    method: str = "cwt",
    window_sec: float = 2.0,
    step_sec: float = 1.0,
    root_dir: Optional[Union[str, Path]] = None,
) -> TimeFrequencyOutputPaths:
    raw, fs = load_signal(signal_data=signal_data, sample_rate=sample_rate, input_file=input_file)
    root = Path(root_dir) if root_dir is not None else create_run_dir(output_dir)

    frames_dir = root / "time_frequency_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    single_file = root / "time_frequency_single.png"

    mono = np.mean(raw, axis=0)
    use_stft = method.lower() == "stft"
    cwt_available = hasattr(signal, "cwt") and hasattr(signal, "morlet2")

    if use_stft:
        freqs, times, amp = compute_stft_map(mono, fs)
        single_title = "STFT Time-Frequency Amplitude Map"
    else:
        freqs, times, amp = compute_cwt_map(mono, fs)
        if cwt_available:
            single_title = "CWT Time-Frequency Amplitude Map"
        else:
            single_title = "STFT Time-Frequency Amplitude Map (CWT Fallback)"

    plot_tf_image(freqs, times, amp, single_file, single_title)

    for idx, start, end in window_indices(mono.shape[0], fs, window_sec, step_sec):
        segment = mono[start:end]
        if use_stft:
            f, t, a = compute_stft_map(segment, fs)
            title = f"STFT Window {idx} ({start/fs:.2f}s-{end/fs:.2f}s)"
        else:
            f, t, a = compute_cwt_map(segment, fs)
            if cwt_available:
                title = f"CWT Window {idx} ({start/fs:.2f}s-{end/fs:.2f}s)"
            else:
                title = f"STFT Window {idx} ({start/fs:.2f}s-{end/fs:.2f}s) (CWT Fallback)"
        plot_tf_image(f, t, a, frames_dir / f"tf_{idx:04d}.png", title)

    return TimeFrequencyOutputPaths(root=root, single_file=single_file, frames_dir=frames_dir)
