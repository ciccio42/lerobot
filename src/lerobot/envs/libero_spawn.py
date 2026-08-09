#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Out-of-distribution spawn-region sampling for LIBERO target objects.

Ported from openvla-oft's `change_bddl_file` (experiments/robot/libero/libero_utils.py), which
rewrites a task's `.bddl` file so the target object's `floor_target_object_region` is replaced
with a region drawn from a `spawn_region.json` file shipped next to the task's bddl files (today
only present for the libero_object suite).

Unlike the openvla-oft version, this writes a uniquely-named file per call (pid + uuid) instead of
a fixed-suffix one, since lerobot may run many vectorized/async envs for the same task concurrently
and a fixed name would let them clobber each other's bddl file.
"""

from __future__ import annotations

import copy
import json
import os
import uuid

import libero.libero.envs.bddl_utils as BDDLUtils
import numpy as np


def sample_spawn_bddl_file(bddl_file: str, spawn_train_distribution: bool) -> str:
    """Write a copy of `bddl_file` with the target object's spawn region resampled.

    Args:
        bddl_file: Path to the task's original `.bddl` file.
        spawn_train_distribution: If True, sample from the "target" region pool (still
            in-distribution, just a different one of the regions seen during training).
            If False, sample from "other_objects" (regions used by other objects in the scene,
            i.e. out-of-distribution for this task's target object).

    Returns:
        Path to the newly written `.bddl` file. The caller owns this file and should delete it
        once the env has been constructed from it.
    """
    bddl_folder = os.path.dirname(bddl_file)
    spawn_region_file = os.path.join(bddl_folder, "spawn_region.json")
    if not os.path.exists(spawn_region_file):
        raise FileNotFoundError(
            f"No spawn_region.json found in {bddl_folder}. Spawn-region randomization is only "
            "supported for LIBERO suites that ship this file (currently libero_object)."
        )
    with open(spawn_region_file) as f:
        spawn_region = json.load(f)

    parsed_bddl = BDDLUtils.robosuite_parse_problem(bddl_file)
    current_spawn_region = parsed_bddl["regions"]["floor_target_object_region"]["ranges"]

    pool = spawn_region["target"] if spawn_train_distribution else spawn_region["other_objects"]
    new_spawn_region = pool[np.random.randint(len(pool))]
    while new_spawn_region == current_spawn_region:
        new_spawn_region = pool[np.random.randint(len(pool))]

    # If another object's region already sits where the target is moving to, swap it into the
    # target's vacated slot so the two objects don't end up spawning on top of each other.
    changed_region_name = None
    for key, region in parsed_bddl["regions"].items():
        if "other_object" in key and region["ranges"] == new_spawn_region:
            changed_region_name = key.split("floor_")[1]
            break

    with open(bddl_file) as f:
        old_bddl_content = f.readlines()
    new_bddl_content = copy.deepcopy(old_bddl_content)

    in_regions_block = False
    for row_idx, row in enumerate(old_bddl_content):
        stripped = row.strip()
        if stripped.startswith("(:regions"):
            in_regions_block = True
        elif stripped.startswith("(:fixtures"):
            in_regions_block = False

        if in_regions_block and "(target_object_region" in row:
            x0, y0, x1, y1 = new_spawn_region[0]
            new_bddl_content[row_idx + 3] = f"              ({x0} {y0} {x1} {y1})\n"
        if in_regions_block and changed_region_name is not None and f"({changed_region_name}" in stripped:
            x0, y0, x1, y1 = current_spawn_region[0]
            new_bddl_content[row_idx + 3] = f"              ({x0} {y0} {x1} {y1})\n"

    suffix = f"_spawn_pid{os.getpid()}_{uuid.uuid4().hex[:8]}.bddl"
    new_bddl_file = bddl_file.replace(".bddl", suffix)
    with open(new_bddl_file, "w") as f:
        f.writelines(new_bddl_content)

    return new_bddl_file
