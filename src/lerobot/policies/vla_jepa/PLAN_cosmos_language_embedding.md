# Plan: Replace VLA-JEPA's language command with a Cosmos-Reason2 embedding

## Question asked

Can the language instruction `ℓ` that VLA-JEPA conditions on be replaced with an embedding
produced by Cosmos-Reason2 (`nvidia/Cosmos-Reason2-8B`) instead of raw tokenized text? **Yes —
and unusually cheaply, because Cosmos-Reason2-8B and VLA-JEPA's own VLM backbone are the same
model class** (`transformers.Qwen3VLForConditionalGeneration`), just different checkpoint sizes
(8B vs 2B). The exact hidden-state-extraction hook VLA-JEPA already uses on its own backbone can
be reused verbatim on Cosmos-Reason2. This plan is written against the actual code in this repo,
not the paper in the abstract — file/line references below are real.

## 1. How the language command is wired in today

VLA-JEPA does **not** have a separate "language embedding `c`" module. The instruction string is
interpolated directly into a text prompt and tokenized by Qwen3-VL's own tokenizer, then the
whole prompt (text + images) goes through one joint transformer forward pass:

- `configuration_vla_jepa.py:52` — the template:
  `"Your task is {instruction}. Infer the temporal dynamics from frames {actions} and produce the
  corresponding policy actions {e_actions}."`
- `qwen_interface.py:79-104` (`Qwen3VLInterface.build_inputs`) — builds a chat-template message
  per sample (images + the filled-in prompt text) and tokenizes it normally via
  `self.processor.apply_chat_template(...)`.
- `modeling_vla_jepa.py:133-160` (`_qwen_last_decoder_hidden`) — runs
  `self.qwen.model(**qwen_inputs, ...)` and grabs the **pre-RMSNorm** hidden state of the last
  decoder layer via a forward hook (needed because `transformers` 5.x's own
  `output_hidden_states=True` returns the post-norm state, which doesn't match how the model was
  trained).
- `modeling_vla_jepa.py:216-237` — gathers the hidden states at the positions of two families of
  **reserved special tokens** already in the vocabulary: `<|action_{i}|>` (feeds the world model)
  and `<|embodied_action|>` (feeds the action head, `num_embodied_action_tokens_per_instruction =
  32` of them). These tokens' final hidden states are the only thing that reaches the action head
  — the instruction only influences the model *indirectly*, through self-attention mixing text and
  image tokens together before those special-token positions read the context back out.

So "the language command" isn't a clean point to intercept — it's diffused into the joint
forward pass. Replacing it with an external embedding means introducing a **new** reserved-token
mechanism, following the exact pattern already used for `<|action_i|>`/`<|embodied_action|>`.

## 2. Two ways to do "replace with a Cosmos embedding" — pick one

**Option A — text substitution (no architecture change, cheap to try first).** Have Cosmos-Reason2
generate/paraphrase the instruction from the demonstration video (this exact call already exists:
`repo/VLA-Benchmark/robosuite_test/vllm_utils.py::run_vllm_server`, used today via
`use_cosmos_name`/`model_cosmos_name` in `run_robosuite_eval.py:33,50,54,166` to swap in a
Cosmos-generated task description). Wire the same call into `VLAJEPAPolicy._prepare_model_inputs`
so `instructions[b]` comes from Cosmos instead of the ground-truth string, before it hits the
*existing* `qwen.build_inputs()`. Zero model changes, ~1 day of work. This is **not** what was
asked (it's not an embedding), but it's the cheapest way to sanity-check whether Cosmos's read of
the scene helps at all before investing in Option B.

**Option B — true embedding-level replacement (what was actually asked).** Compute a fixed-size
vector from Cosmos-Reason2, project it into Qwen3-VL's embedding space, and inject it as the
hidden state backing a new block of reserved tokens — bypassing Qwen3-VL's own tokenizer/embedding
lookup for the instruction. The rest of this plan implements Option B.

## 3. Option B, step by step

### Step 1 — Extract a Cosmos-Reason2 embedding

Reuse `Qwen3VLInterface` almost as-is against the 8B checkpoint instead of the 2B one:

```python
cosmos = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
    "nvidia/Cosmos-Reason2-8B", torch_dtype=torch.bfloat16
)
processor = transformers.Qwen3VLProcessor.from_pretrained("nvidia/Cosmos-Reason2-8B")
```

(This is literally the same call already made in
`repo/Video-Captioning/Video-Captioning-Human-Demo/cosmos-reason2/scripts/inference_sample.py:45,54`
— confirms Cosmos-Reason2-8B is a `Qwen3VLForConditionalGeneration` checkpoint, not a separate
architecture.)

Build a prompt from the instruction (optionally + the first frame, for a visually-grounded
embedding rather than a text-only one — see the open question in §5), run a forward pass with the
same forward-hook trick as `_qwen_last_decoder_hidden`, and **mean-pool** the last-layer hidden
states over the instruction's token span (mean-pooling is more standard than last-token pooling
for sentence-level embeddings from a decoder LM, and doesn't require locating an EOS position).
Output: one vector of size `cosmos_hidden_dim` per instruction.

**Verified, not guessed:** `nvidia/Cosmos-Reason2-8B` is already downloaded on this cluster at
`/home/rsofnc000/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-8B` (19GB, no download
needed). Its `config.json` confirms `model_type: "qwen3_vl"` and `text_config.hidden_size: 4096` —
i.e. `cosmos_embedding_dim = 4096`. VLA-JEPA's own `self.qwen.model.config.hidden_size`
(Qwen3-VL-2B-Instruct's text hidden size) still needs confirming at runtime — it wasn't cached
locally to check directly, and shouldn't be assumed equal to Cosmos's 4096 given the size
difference (2B vs 8B).

### Step 2 — Cache, don't recompute live

Cosmos-Reason2-8B is a large model; running it as an extra forward pass on every training step
alongside the 2B backbone + V-JEPA2 encoder would roughly double VRAM/compute for training. Given
robot instructions repeat heavily across episodes (this project's UR5e pick-place tasks have a
small fixed set of `command.json` strings per variation), **precompute embeddings once per unique
instruction string and cache to disk** (a `{instruction_hash: embedding}` file, analogous to how
`dataset_statistics_*.json` is precomputed once per dataset rather than at train time). At train
and inference time, look the cached vector up by instruction string — no Cosmos-Reason2 weights
need to be loaded into the training process at all. Only fall back to a live Cosmos-Reason2 call
for genuinely novel instructions at eval time (OOD language robustness testing).

### Step 3 — New reserved token span + projector

In `configuration_vla_jepa.py`, add:

```python
cosmos_lang_token: str = "<|cosmos_lang_{}|>"
num_cosmos_lang_tokens: int = 8          # tunable; parallels num_embodied_action_tokens_per_instruction
cosmos_embedding_dim: int = 4096         # verified: Cosmos-Reason2-8B's text_config.hidden_size
use_cosmos_language_embedding: bool = False   # feature flag, default off
```

In `qwen_interface.py::expand_tokenizer()`, register `num_cosmos_lang_tokens` new special tokens
the same way `<|action_i|>`/`<|embodied_action|>` are registered (lines 55-77), and resize the
embedding table.

Add a small trainable projector in `modeling_vla_jepa.py::VLAJEPAModel.__init__`:

```python
if config.use_cosmos_language_embedding:
    self.cosmos_lang_projector = nn.Linear(config.cosmos_embedding_dim, self.qwen.model.config.hidden_size)
```

A single `Linear` is the minimum viable choice; a 2-layer MLP with a GELU is a reasonable upgrade
if the linear version underfits — treat this as a hyperparameter, not a fixed design choice.

### Step 4 — Splice the projected embedding into `inputs_embeds`

This is the part that needs care. Do **not** try to hand-build the whole input sequence's
embeddings from scratch — HF's `Qwen3VLForConditionalGeneration.forward()` still needs `input_ids`
present to locate image-token placeholder positions internally (this is the same constraint that
`open-pi-zero`'s own vision-embedding injection works around — see
`open-pi-zero/src/model/vla/pizero.py::_forward_siglip_and_text_embedding`, and
`repo/lerobot/lerobot/src/lerobot/policies/wall_x/modeling_wall_x.py:731-860`, which does the same
`masked_scatter`-based row-overwrite for image embeddings in a Qwen2.5-VL-family model — the
closest existing precedent in this repo for exactly this pattern, just for a different modality).

The correct sequence, added as a new method on `Qwen3VLInterface` (e.g. `build_inputs_cosmos`):

1. Replace `{instruction}` in the prompt template with `num_cosmos_lang_tokens` repetitions of
   `cosmos_lang_token`, instead of the raw instruction text. Tokenize normally via
   `apply_chat_template` — this gives correct `input_ids`/`attention_mask`/`pixel_values` with
   placeholder rows sitting where the instruction used to be.
2. Run the model's normal embedding lookup to get the full `inputs_embeds` tensor:
   `inputs_embeds = self.model.get_input_embeddings()(input_ids)`.
3. Locate the cosmos-lang-token positions the same way `embodied_mask`/`action_mask` are located
   in `modeling_vla_jepa.py:217-226` (`input_ids == cosmos_lang_token_ids`).
4. Project the cached Cosmos embedding (Step 1/2) through `self.cosmos_lang_projector`, and
   overwrite those specific rows of `inputs_embeds` with the projected vector (broadcast across
   the `num_cosmos_lang_tokens` positions, or split the projector output across them — either is
   fine as a starting point).
5. Call `self.qwen.model(inputs_embeds=inputs_embeds, attention_mask=..., pixel_values=...,
   image_grid_thw=..., output_hidden_states=False, ...)` instead of passing `input_ids`. Confirm
   via a quick unit test that `Qwen3VLForConditionalGeneration.forward` accepts `inputs_embeds`
   together with `pixel_values` (it should, since HF's own image-embedding merge internally uses
   this same input path) — this is the single riskiest unverified assumption in this plan and
   should be the very first thing checked, before writing any of the surrounding code.
6. `_qwen_last_decoder_hidden` (`modeling_vla_jepa.py:133`) needs no changes — the forward hook
   only cares about the layer's output, not how the model was invoked.

### Step 5 — Training

This is a real architecture change, not a config toggle on a frozen checkpoint:

- The pretrained `outputs/ur5e_vla_jepa` checkpoint was never trained with this input
  distribution at the instruction-token positions, so **loading it and evaluating cold will not
  work**. At minimum, fine-tune from that checkpoint with `reinit_modules` covering the new
  projector's parameter names (see the existing shape-mismatch-tolerant loading path in
  `modeling_vla_jepa.py:593-629`, `_load_as_safetensor`).
- Keep Cosmos-Reason2-8B itself **fully frozen** (never load its weights into the training job at
  all if Step 2's caching is in place) — training it is out of scope and unnecessary; only the new
  projector (and optionally a LoRA adapter on the Qwen3-VL-2B backbone, mirroring
  `freeze_qwen`/existing LoRA patterns elsewhere in this project) need gradient updates.
- Reuse `repo/lerobot/lerobot/run_train_scripts/train_ur5e_vla_jepa.sh` as the base training
  script, adding `--policy.use_cosmos_language_embedding=true` once the flag exists.

### Step 6 — Evaluation

No new eval infrastructure needed — `repo/VLA-Benchmark/robosuite_test/run_lerobot_vla_jepa.sh` /
`models/lerobot_vla_jepa_eval_config.yml` already exercise this exact policy end-to-end. Run an
A/B: the existing raw-text checkpoint vs. the Cosmos-embedding fine-tune, same task suite, same
`num_trials_per_task`, and compare success/picked/reached rates as already logged in
`LEROBOT_EVAL.md`'s format. The periodic-checkpoint-eval-with-Telegram-reporting pattern built
this session for Interleave-VLA (`Interleave-VLA/open-pi-zero/slurm/autoresume_watch.sh`'s
`check_periodic_eval`) is a directly reusable template if this fine-tune is left training
unattended for multiple days.

## 4. Honest expectations

`LEROBOT_EVAL.md` §6 ("Why VLA-JEPA picks but never places") already root-caused VLA-JEPA's
current main failure mode as an **undertrained gripper-release sub-behavior** (1.17 epochs, loss
still plateauing) plus a since-fixed harness asymmetry — not a language-understanding problem.
Swapping in a Cosmos-Reason2 embedding is a bet on **generalization/robustness to instruction
phrasing** (the kind of gain VLA-JEPA's own paper shows for the "Language" perturbation column in
LIBERO-Plus, Table 3), not a fix for the placement failure mode already diagnosed in this repo.
Frame the experiment accordingly — the two are independent axes of improvement.

## 5. Open questions to resolve before/while implementing

1. **Text-only vs. text+image Cosmos embedding.** A text-only embedding is far cheaper to cache
   (one entry per unique instruction string) but throws away Cosmos's actual visual grounding
   ability. A text+first-frame embedding is per-episode, not per-instruction, so the cache key and
   storage cost change accordingly. Start text-only; revisit if results are disappointing.
2. **Pooling strategy** (mean over instruction tokens vs. last-token vs. a fixed number of
   evenly-sampled positions) — untested, worth a small ablation once the pipeline works end to end.
3. **`inputs_embeds` + `pixel_values` compatibility** — flagged above as the single technical risk
   that should be verified with a 10-line standalone script before any other code is written.
4. **Exact hidden dimensions** for both models — verify via `AutoConfig`, don't hardcode from this
   document.

## 6. File change checklist

- `configuration_vla_jepa.py` — new config fields (§3, Step 3).
- `qwen_interface.py` — extend `expand_tokenizer()`; add `build_inputs_cosmos()`.
- `modeling_vla_jepa.py` — add `cosmos_lang_projector`; branch `forward`/`predict_action` on
  `config.use_cosmos_language_embedding` to call `build_inputs_cosmos()` instead of `build_inputs()`.
- New file, e.g. `cosmos_embedding_cache.py` — offline extraction script (Step 1) + cache
  load/lookup helper used by both training and inference.
- `run_train_scripts/train_ur5e_vla_jepa.sh` (or a copy) — add the new CLI flag.
- No changes needed in `processor_vla_jepa.py`, `world_model.py`, `action_head.py`, or anywhere in
  `VLA-Benchmark/` — the embedding swap is entirely upstream of the parts those touch.
