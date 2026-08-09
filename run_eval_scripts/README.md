# LIBERO eval scripts

SLURM (`sbatch`) wrappers around `lerobot-eval` for preliminary LIBERO
experiments. All scripts `cd` into the repo root (`lerobot/lerobot/`, the
dir with `pyproject.toml`) before calling `srun lerobot-eval`, so they can
be submitted from anywhere.

## Prerequisites

- These scripts run in the `lerobot` conda env
  (`/mnt/beegfs/frosa/.conda/envs/lerobot`), activated inside each script
  itself (`conda activate lerobot`) — you don't need to activate anything
  before `sbatch`.
- LIBERO extra installed in that env: `pip install -e ".[libero]"`. This
  pulls in `hf-libero` (a PyPI-packaged, pip-installable LIBERO — bddl
  files, init states, envs — importable as `libero.*`), which is why no
  `PYTHONPATH` pointing at a separate cloned LIBERO repo is needed anymore.
  - **Building it needs `CMAKE_POLICY_VERSION_MINIMUM=3.5` set** (and
    `--no-build-isolation` if the ambient `cmake` isn't on `PATH` inside
    pip's isolated build env): `hf-egl-probe`/`egl_probe`'s bundled
    `CMakeLists.txt` declares `cmake_minimum_required(VERSION 2.8.12)`,
    which CMake ≥4.0 (installed here via the `cmake` pip package) refuses
    to configure without that policy override. Full command that's known
    to work in this env:
    ```bash
    PATH="/mnt/beegfs/frosa/.conda/envs/lerobot/bin:$PATH" \
      CMAKE_POLICY_VERSION_MINIMUM=3.5 \
      /mnt/beegfs/frosa/.conda/envs/lerobot/bin/pip install --no-build-isolation -e ".[libero]"
    ```
  - `hf-libero` pins `mujoco<3.9.0,>=3.0.0` — **do not let this drift above
    3.8.x**. `robosuite==1.4.0`'s `controllers/base_controller.py` calls
    `self.sim.data.qM`, an attribute MuJoCo's Python bindings dropped
    somewhere after 3.8.x (replaced by `.M`); anything ≥3.9 crashes with
    `AttributeError: 'MjData' object has no attribute 'qM'` on env reset.
    Installing via the `[libero]` extra (rather than `pip install
    robosuite`/`mujoco` directly) keeps this pin honored automatically.
- `~/.libero/config.yaml` present (see `docker/Dockerfile.benchmark.libero`
  for the expected format) so the `libero` package can locate assets,
  bddl files, and init states.
- `MUJOCO_GL=egl` is exported inside each script for headless rendering on
  the cluster — no need to set it yourself.
- `lerobot/pi0_libero`, `lerobot/pi0fast-libero` etc. are public HF models;
  if you swap in a gated/private checkpoint (e.g. `google/paligemma-3b-pt-224`
  is gated), run `huggingface-cli login` first. Note each script sets its own
  `HF_HOME`/`HUGGINGFACE_HUB_CACHE` (see script) — your login token needs to
  exist at `${HF_HOME}/token`, not just the default `~/.cache/huggingface/token`.

## Scripts

### 1. `run_libero_pi0.sh` — classic LIBERO, pi0

Evaluates `lerobot/pi0_libero` across the four standard suites
(Spatial, Object, Goal, Long) at 10 episodes/task — the protocol used for
LeRobot's published results.

```bash
sbatch run_libero_pi0.sh
```

Override via env vars:

```bash
POLICY_PATH=lerobot/pi0_libero_finetuned TASKS=libero_object N_EPISODES=5 \
  sbatch run_libero_pi0.sh
```

### 2. `run_libero_pi0fast.sh` — classic LIBERO, pi0fast

Same protocol, using `lerobot/pi0fast-libero`. pi0fast expects
`observation.images.{base_0_rgb,left_wrist_0_rgb}` while the LIBERO env
returns `observation.images.{image,image2}`, so the script passes a
`--rename_map` to bridge the two (see `docs/source/rename_map.mdx`).

```bash
sbatch run_libero_pi0fast.sh
```

Same override variables as above (`POLICY_PATH`, `TASKS`, `N_EPISODES`,
`BATCH_SIZE`, `OUTPUT_DIR`).

### 3. `run_libero_object_spawn.sh` — customized libero_object, spawn randomization

Ports the spawn-region randomization used in
`openvla-oft/experiments/robot/libero/run_libero_eval.sh` /
`run_libero_eval.py` (`--change_spawn True --spawn_train_distribution False`)
to LeRobot's LIBERO env, which implements the identical mechanism natively
via `--env.change_spawn` / `--env.spawn_train_distribution`. Only
`libero_object` ships a `spawn_region.json`, so this only targets that
suite. Every episode rebuilds the MuJoCo scene from a bddl file with a
freshly-sampled `floor_target_object_region`, so it runs noticeably slower
than the classic scripts above.

Takes the policy family as a positional arg (`pi0` default, or `pi0fast`):

```bash
sbatch run_libero_object_spawn.sh pi0
sbatch run_libero_object_spawn.sh pi0fast
```

`--env.spawn_train_distribution=false` (the default here, matching
openvla-oft) samples an out-of-distribution spawn region; pass
`SPAWN_TRAIN_DISTRIBUTION=true` for the in-distribution variant:

```bash
SPAWN_TRAIN_DISTRIBUTION=true sbatch run_libero_object_spawn.sh pi0
```

Other overrides: `POLICY_PATH`, `N_EPISODES` (default 10), `BATCH_SIZE`
(default 2), `OUTPUT_DIR`.

## Notes

- `--env.max_parallel_tasks=1` is set in the classic scripts to match the
  recommended/reproduction commands in `docs/source/libero.mdx`.
- Results and logs are written under `--output_dir` (defaults to
  `./eval_logs/<name>` relative to the repo root, i.e.
  `lerobot/lerobot/eval_logs/...`).
- For a from-scratch multi-suite pi0.5 reproduction command (not wrapped
  here yet), see the "Reproducing published results" section of
  `docs/source/libero.mdx`.
