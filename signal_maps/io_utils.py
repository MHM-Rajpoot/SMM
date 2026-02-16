from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union
import shutil

import numpy as np
from scipy.io import wavfile

ArrayLike = Union[np.ndarray, Sequence[float]]


def ensure_2d(signal_data: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal_data, dtype=np.float64)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError("signal_data must be 1D or 2D with shape (channels, samples).")
    return arr


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    max_abs = np.max(np.abs(x)) + 1e-12
    return x / max_abs


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        fs, data = wavfile.read(path)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            return data[np.newaxis, :], int(fs)
        return data.T, int(fs)

    if suffix == ".mp3":
        try:
            import librosa  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "MP3 input requires librosa. Install with: pip install librosa soundfile"
            ) from exc

        data, fs = librosa.load(path.as_posix(), sr=None, mono=False)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            return data[np.newaxis, :], int(fs)
        return data, int(fs)

    raise ValueError("Only .wav and .mp3 are supported for file-based audio input.")


def load_signal(
    signal_data: Optional[ArrayLike] = None,
    sample_rate: Optional[int] = None,
    input_file: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, int]:
    if input_file is None and signal_data is None:
        raise ValueError("Provide either input_file or signal_data + sample_rate.")

    if input_file is not None:
        raw, fs = load_audio(Path(input_file))
    else:
        if sample_rate is None:
            raise ValueError("sample_rate is required when using in-memory signal_data.")
        raw = ensure_2d(np.asarray(signal_data, dtype=np.float64))
        fs = int(sample_rate)

    return normalize(raw), fs


def create_run_dir(
    base_output_dir: Union[str, Path],
    use_timestamp: bool = True,
    overwrite: bool = False,
) -> Path:
    if use_timestamp:
        stamp = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        root = Path(base_output_dir) / stamp
    else:
        root = Path(base_output_dir)

    if overwrite and root.exists():
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except PermissionError:
                    pass

    root.mkdir(parents=True, exist_ok=True)
    return root


def window_indices(n_samples: int, fs: int, window_sec: float, step_sec: float) -> Iterable[Tuple[int, int, int]]:
    w = max(1, int(window_sec * fs))
    step = max(1, int(step_sec * fs))
    idx = 0
    for start in range(0, max(1, n_samples - w + 1), step):
        end = min(start + w, n_samples)
        if end - start < max(4, w // 3):
            continue
        yield idx, start, end
        idx += 1
