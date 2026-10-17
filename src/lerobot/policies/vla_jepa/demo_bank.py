# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_instruction(text: str) -> str:
    return " ".join(text.strip().rstrip(".").lower().split())


class DemoVideoBank:
    """Demonstration videos (e.g. human demos) indexed by task, used to condition VLA-JEPA.

    Layout: ``<root>/<task_folder>/*.mp4`` (sorted by name). The demo shown for a sample is always a
    video of a *different agent / episode* than the robot trajectory being trained on, so it never
    contains the future frames the world model has to predict (no leakage).

    The last ``num_eval_per_task`` videos of each task are held out for evaluation.
    """

    def __init__(
        self,
        root: str | Path,
        task_map_path: str | Path,
        num_frames: int,
        image_size: int,
        num_eval_per_task: int = 8,
        cache_videos: bool = True,
    ) -> None:
        self.root = Path(root)
        self.num_frames = num_frames
        self.image_size = image_size
        self.cache_videos = cache_videos
        self._cache: dict[Path, tuple[np.ndarray, float]] = {}
        self._eval_counters: dict[str, int] = {}
        self.last_sample: dict | None = None  # which video / frames the last sample() returned

        with open(task_map_path) as f:
            raw_map = json.load(f)
        self.task_map = {normalize_instruction(k): v for k, v in raw_map.items()}

        self.train_videos: dict[str, list[Path]] = {}
        self.eval_videos: dict[str, list[Path]] = {}
        for folder in sorted(set(self.task_map.values())):
            videos = sorted((self.root / folder).glob("*.mp4"))
            if len(videos) <= num_eval_per_task:
                raise ValueError(
                    f"Task folder {self.root / folder} has {len(videos)} videos, need more than "
                    f"demo_num_eval_per_task={num_eval_per_task}."
                )
            split = len(videos) - num_eval_per_task
            self.train_videos[folder] = videos[:split]
            self.eval_videos[folder] = videos[split:]

    def task_folder(self, instruction: str) -> str:
        key = normalize_instruction(instruction)
        if key not in self.task_map:
            raise KeyError(f"No demo task mapped to instruction {instruction!r} (check demo_task_map_path).")
        return self.task_map[key]

    def _load(self, path: Path) -> tuple[np.ndarray, float]:
        if path in self._cache:
            return self._cache[path]
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 10.0
            frames = [
                np.asarray(
                    frame.to_image().resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                )
                for frame in container.decode(video=0)
            ]
        video = np.stack(frames)  # [L, S, S, 3] uint8, squashed (not cropped) to keep the whole scene
        if self.cache_videos:
            self._cache[path] = (video, fps)
        return video, fps

    def _frame_indices(self, length: int, rng: np.random.Generator | None) -> np.ndarray:
        # TSN-style: split the whole demo into num_frames segments, one frame per segment
        # (random in training, segment center in eval).
        edges = np.linspace(0, length, self.num_frames + 1)
        if rng is None:
            idx = (edges[:-1] + edges[1:]) / 2
        else:
            idx = edges[:-1] + rng.random(self.num_frames) * (edges[1:] - edges[:-1])
        return np.clip(idx.astype(int), 0, length - 1)

    def sample(
        self, instruction: str, training: bool, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, float]:
        """Return (frames [num_frames, S, S, 3] uint8, effective fps of the sampled frames)."""
        folder = self.task_folder(instruction)
        if training:
            rng = rng if rng is not None else np.random.default_rng()
            videos = self.train_videos[folder]
            path = videos[rng.integers(len(videos))]
        else:
            # Cycle through the held-out demos so consecutive eval episodes see different demos.
            videos = self.eval_videos[folder]
            count = self._eval_counters.get(folder, 0)
            self._eval_counters[folder] = count + 1
            path = videos[count % len(videos)]
            rng = None
        video, fps = self._load(path)
        idx = self._frame_indices(len(video), rng)
        self.last_sample = {"video": str(path), "task_folder": folder, "frame_indices": idx.tolist()}
        effective_fps = self.num_frames / (len(video) / fps)
        return video[idx], effective_fps
