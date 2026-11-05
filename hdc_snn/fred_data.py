"""FRED event-frame window loader for sequence/scene recognition.

FRED's raw event ``.hdf5`` requires the proprietary Prophesee/Metavision SDK,
so we use the rendered *event frames* (PNG) shipped in each sequence zip. Each
recording sequence is treated as one "place"/scene, and the task is to
recognise which scene a temporal window of event frames came from.

Frames are pre-resized to a small grayscale resolution and cached in memory so
epochs remain fast, and each scene is split in time to keep train/test windows
disjoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _frame_timestamp(path: str) -> int:
    name = os.path.basename(path)
    # names look like "Video_21_frame_100032333.png" (microseconds)
    if "frame_" in name:
        digits = name.split("frame_")[-1].split(".")[0]
        try:
            return int(digits)
        except ValueError:
            pass
    return int("".join(ch for ch in name if ch.isdigit()) or "0")


class FREDFrameWindowDataset(Dataset):
    """Temporal windows of consecutive event frames with a scene label.

    Each item is ``(window[T, 1, H, W] float in [0,1], scene_id)``.
    """

    def __init__(
        self,
        root: str,
        split: str,
        ids: list[str],
        window: int = 8,
        stride: int = 4,
        size: int = 64,
        train_frac: float = 0.7,
    ) -> None:
        self.window = window
        self.size = size
        # (start_index, frame_array, scene_id)
        samples: list[tuple[int, np.ndarray, int]] = []

        for scene_idx, seq_id in enumerate(ids):
            frames_dir = os.path.join(
                root, "extracted", split, seq_id, seq_id, "Event", "Frames"
            )
            if not os.path.isdir(frames_dir):
                raise FileNotFoundError(f"FRED frames not found: {frames_dir}")
            paths = sorted(
                [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")],
                key=_frame_timestamp,
            )
            if len(paths) < window:
                continue
            frames = self._load_sequence(paths)
            n = frames.shape[0]
            for start in range(0, n - window + 1, stride):
                center = start + window // 2
                is_train = center < train_frac * n
                if is_train == (split == "train"):
                    samples.append((start, frames, scene_idx))

        self.samples = samples

    def _load_sequence(self, paths: list[str]) -> np.ndarray:
        """Resize to grayscale ``[F, size, size]`` uint8."""
        out = np.empty((len(paths), self.size, self.size), dtype=np.uint8)
        for i, p in enumerate(paths):
            img = Image.open(p).convert("L").resize(
                (self.size, self.size), Image.Resampling.BILINEAR
            )
            out[i] = np.asarray(img, dtype=np.uint8)
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start, frames, scene = self.samples[idx]
        x = frames[start : start + self.window].astype(np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(1)  # [T, 1, H, W]
        return x, torch.tensor(scene, dtype=torch.long)


@dataclass
class FREDSplit:
    train: FREDFrameWindowDataset
    test: FREDFrameWindowDataset


def make_fred_split(
    root: str,
    train_ids: list[str],
    test_ids: list[str],
    window: int = 8,
    stride: int = 4,
    size: int = 64,
    train_frac: float = 0.7,
) -> FREDSplit:
    train = FREDFrameWindowDataset(
        root, "train", train_ids, window=window, stride=stride, size=size, train_frac=train_frac
    )
    test = FREDFrameWindowDataset(
        root, "test", test_ids, window=window, stride=stride, size=size, train_frac=train_frac
    )
    return FREDSplit(train, test)