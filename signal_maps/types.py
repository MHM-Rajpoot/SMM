from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TimeFrequencyOutputPaths:
    root: Path
    single_file: Path
    frames_dir: Path


@dataclass
class TopomapOutputPaths:
    root: Path
    single_file: Path
    frames_dir: Path


@dataclass
class CombinedOutputPaths:
    root: Path
    time_frequency: TimeFrequencyOutputPaths
    topomap: TopomapOutputPaths
    combined_single_file: Path
    combined_frames_file: Path
    combined_frames_dir: Path
