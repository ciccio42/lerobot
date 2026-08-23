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

Evaluates `lerobot/pi0_libero_finetuned_v044` across the four standard suites
(Spatial, Object, Goal, Long) at 10 episodes/task — the protocol used for
LeRobot's published results. Its `input_features` already use
`observation.images.{image,image2}`, matching the LIBERO env natively, so
no `--rename_map` is needed.

```bash
sbatch run_libero_pi0.sh
```

Override via env vars:

```bash
POLICY_PATH=lerobot/pi0_libero_finetuned TASKS=libero_object N_EPISODES=5 \
  sbatch run_libero_pi0.sh
```

### 2. `run_libero_pi0fast.sh` — classic LIBERO, pi0fast

Same protocol, using `lerobot/pi0fast-libero-v044`. pi0fast expects
`observation.images.{base_0_rgb,left_wrist_0_rgb}` (plus a policy-side
`empty_camera_0` padding slot from `empty_cameras=1`, which the env doesn't
need to supply — see `validate_visual_features_consistency`) while the
LIBERO env returns `observation.images.{image,image2}`, so the script passes
a `--rename_map` to bridge the two (see `docs/source/rename_map.mdx`).

- **`jadechoghari/fast-libero-tokenizer-mean-std`'s action tokenizer fails to
  load via plain `AutoProcessor.from_pretrained(..., trust_remote_code=True)`**
  with a misleading `... install sentencepiece or tiktoken ...` error, in
  *two* call sites: `PI0FastPolicy.__init__` (`modeling_pi0_fast.py`) and
  `ActionTokenizerProcessorStep.__post_init__`
  (`processor/tokenizer_processor.py`). Root cause: a `transformers`
  `ProcessorMixin` bug — its generic sub-component loader only treats a
  tokenizer attribute as "primary" (loaded from the repo root) when it's
  named exactly `tokenizer`; this repo's custom processor names its
  attribute `bpe_tokenizer`, so `transformers` looks for it in a
  `bpe_tokenizer/` subfolder that doesn't exist, and falls through to a slow→
  fast conversion path that also fails. Both call sites now fall back to a
  `_load_action_tokenizer()` helper that loads the tokenizer and processor
  class manually (present in both files) when the primary path raises
  `ValueError`. Only bites `action_tokenizer_name` repos using this
  `bpe_tokenizer`-attribute pattern; harmless/unreachable otherwise.

```bash
sbatch run_libero_pi0fast.sh
```

Same override variables as above (`POLICY_PATH`, `TASKS`, `N_EPISODES`,
`BATCH_SIZE`, `OUTPUT_DIR`).

### 3. `run_libero_pi05.sh` — classic LIBERO, pi0.5

Same protocol, using `lerobot/pi05_libero_finetuned_v044` (the checkpoint
`docs/source/pi05.mdx`'s "Reproducing published results" section refers to
as `pi05_libero_finetuned` — same repo, both names resolve to it). Like pi0,
its `input_features` already use `observation.images.{image,image2}`
natively and it doesn't use an external action tokenizer, so no
`--rename_map` is needed.

```bash
sbatch run_libero_pi05.sh
```

Same override variables as above (`POLICY_PATH`, `TASKS`, `N_EPISODES`,
`BATCH_SIZE`, `OUTPUT_DIR`).

### 4. `run_libero_molmoact2.sh` — classic LIBERO, MolmoAct2

Evaluates `allenai/MolmoAct2-LIBERO-LeRobot`, following the "Evaluation With
LeRobot MolmoAct2 Weight" recipe in `docs/source/molmoact2.mdx` exactly
(`policy.inference_action_mode=continuous`, `bfloat16`+AMP, CUDA-graph
inference, per-episode seeding, and `--env.camera_name_mapping` to map LIBERO's
raw camera names — MolmoAct2 expects `image`/`wrist_image`, not LeRobot's own
`image`/`image2` convention). Large model (~7B-class VLM backbone); expect
several minutes just to download/load weights before rollouts start.

```bash
sbatch run_libero_molmoact2.sh
```

Same override variables as above (`POLICY_PATH`, `TASKS`, `N_EPISODES`,
`BATCH_SIZE`, `OUTPUT_DIR`).

### 5. `run_libero_vla_jepa.sh` — classic LIBERO, VLA-JEPA

Evaluates `lerobot/VLA-JEPA-LIBERO` (a JEPA world-model policy), following
the "Reproducing the LIBERO results" recipe in `docs/source/vla_jepa.mdx`.
No `--rename_map` needed. Note the doc's reference command uses
`--eval.batch_size=5` (not 1) — kept as this script's default.

```bash
sbatch run_libero_vla_jepa.sh
```

Same override variables as above (`POLICY_PATH`, `TASKS`, `N_EPISODES`,
`BATCH_SIZE`, `OUTPUT_DIR`).

### 6. `run_libero_object_spawn.sh` — customized libero_object, spawn randomization

Ports the spawn-region randomization used in
`openvla-oft/experiments/robot/libero/run_libero_eval.sh` /
`run_libero_eval.py` (`--change_spawn True --spawn_train_distribution False`)
to LeRobot's LIBERO env, which implements the identical mechanism natively
via `--env.change_spawn` / `--env.spawn_train_distribution`. Only
`libero_object` ships a `spawn_region.json`, so this only targets that
suite. Every episode rebuilds the MuJoCo scene from a bddl file with a
freshly-sampled `floor_target_object_region`, so it runs noticeably slower
than the classic scripts above.

Takes the policy family as a positional arg (`pi0` default, `pi0fast`,
`pi05`, `molmoact2`, or `vla_jepa`):

```bash
sbatch run_libero_object_spawn.sh pi0
sbatch run_libero_object_spawn.sh pi0fast
sbatch run_libero_object_spawn.sh pi05
sbatch run_libero_object_spawn.sh molmoact2
sbatch run_libero_object_spawn.sh vla_jepa
```

`--env.spawn_train_distribution=false` (the default here, matching
openvla-oft) samples an out-of-distribution spawn region; pass
`SPAWN_TRAIN_DISTRIBUTION=true` for the in-distribution variant:

```bash
SPAWN_TRAIN_DISTRIBUTION=true sbatch run_libero_object_spawn.sh pi0
```

`OUTPUT_DIR` defaults to
`./eval_logs/libero_object_spawn_<policy>_<train_dist|ood_dist>`, so both
distributions for the same policy get separate directories automatically.

Other overrides: `POLICY_PATH`, `N_EPISODES` (default 10), `BATCH_SIZE`
(per-policy default: 2 for pi0/pi0fast/pi05, 1 for molmoact2, 5 for
vla_jepa), `OUTPUT_DIR`.

- **This script was missing `conda activate lerobot` and `--exclude=gnode09`**
  (present in all the other scripts in this dir) — fixed; without them it
  failed immediately with `lerobot-eval: No such file or directory` or a
  SLURM `TaskProlog failed` on that node.
- **Spawn randomization silently had no effect** because `LiberoEnv`
  defaults `init_states=True`, and neither this script nor
  `docs/source/libero.mdx`'s own example set `--env.init_states=false`. Every
  reset rebuilt the sim from a freshly-resampled bddl file (correctly moving
  the target object), but then immediately called `set_init_state()` on the
  fixed, recorded demo state — which overwrites the *entire* sim state,
  object positions included, undoing the randomization. Fixed in
  `src/lerobot/envs/libero.py`'s `LiberoEnv.reset()`: `set_init_state()` is
  now skipped whenever `change_spawn=True`. Verified by resetting a
  `libero_object` task 4 times with `change_spawn=True` and reading back the
  target object's body position from `sim.data.get_body_xpos()` each time —
  it now actually varies between resets. Any spawn-eval results produced
  before this fix (including this script's earlier runs) never tested spawn
  generalization at all — they silently ran the fixed default layout every
  episode and should be disregarded/rerun.

## Notes

- `--env.max_parallel_tasks=1` is set in the classic scripts to match the
  recommended/reproduction commands in `docs/source/libero.mdx`.
- Results and logs are written under `--output_dir` (defaults to
  `./eval_logs/<name>` relative to the repo root, i.e.
  `lerobot/lerobot/eval_logs/...`).
- Set `LEROBOT_EVAL_DEBUG_IMAGES=1` (env var, off by default) before
  `sbatch`-ing any of these to dump the policy's preprocessed camera inputs
  as PNGs under `<output_dir>/debug_images/<suite>_<task_id>/batch<N>/` —
  useful for sanity-checking what the policy actually sees. See the
  `debug_images_dir` plumbing in `scripts/lerobot_eval.py`.
