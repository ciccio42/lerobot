# VLA-JEPA condizionato da video dimostrativo

Obiettivo: rendere VLA-JEPA (arXiv 2602.10098) **video-conditioned**. Il comando testuale ℓ delle
Eq. 2 e 6 viene sostituito da un **video dimostrativo umano** dello stesso task, girato da un altro
agente e in un altro episodio rispetto alla traiettoria robotica su cui si calcolano le azioni.
È lo stesso setting del Multi-Task-LFD-Training-Framework (`demo_name: human_rgb`, `agent_name: ur5e`).

## 1. Formulazione

Nel codice le Eq. 2 e 6 corrispondono a **un solo forward di Qwen3-VL con un solo prompt**. Per
questo sostituire ℓ nell'Eq. 6 la sostituisce anche nell'Eq. 2. Con **d** = embedding della demo:

- z_{t_i} = p_VLM( ⟨latent_i⟩ | **d**, {I_{j,t0}} )           (Eq. 2 modificata, latent action per il world model)
- z_a     = p_VLM( ⟨action⟩ | **d**, {I_{j,t0}}, ⟨latent_i⟩ )  (Eq. 6 modificata, condizionamento della action head)

Le loss non cambiano: L = L_FM + β·L_WM (Eq. 9). Il gradiente di entrambe le loss arriva ai token
demo attraverso la self-attention del LLM. È così che il modello impara a estrarre dalla demo quale
task eseguire.

Prompt con la demo (`demo_prompt_template`):

```
[img I_t0 vista 1][img I_t0 vista 2] The demonstration {DEMO} shows your task. Infer the temporal dynamics
from frames <|action_0|>×8 … <|action_2|>×8 and produce the corresponding policy actions <|embodied_action|>×32.
```

### Due encoder della demo

| | Opzione A: `demo_encoder=vjepa` (metodo proposto) | Opzione C: `demo_encoder=qwen_video` (baseline) |
|---|---|---|
| Encoder | V-JEPA2 ViT-L **congelato** (lo stesso che fornisce i target del WM) | Vision encoder di Qwen3-VL (SigLIP-2), input video nativo |
| Frame demo | 16, 256×256 → 8×256 = 2048 token × 1024 | 16, 224×224 → 8×49 = 392 token video |
| Adattatore | `DemoResampler` (Perceiver, 2 blocchi, 64 query, ~36M param, **nuovo**) | nessuno |
| In `{DEMO}` | `<\|demo\|>`×64: gli embedding vengono sostituiti dall'output del resampler tramite hook su `embed_tokens` | i token video di Qwen, con timestamp reali |
| Idea | la demo entra nello **stesso spazio di dinamica** del world model | la demo entra in uno spazio semantico già noto al LLM |

Sono comuni alle due opzioni:
- L'osservazione robotica corrente passa sempre dal vision encoder di Qwen.
- La clip futura del robot passa da V-JEPA2 ed è **solo** il target di L_WM.

**Nessun leakage:** la demo è un video umano di un altro episodio. Non contiene mai i frame futuri
che il world model deve predire. Se si usasse come demo lo stesso episodio robotico, L_WM
collasserebbe, cioè si ricreerebbe proprio la scorciatoia che il paper elimina.

## 2. Implementazione

| File | Modifica |
|---|---|
| `src/lerobot/policies/vla_jepa/configuration_vla_jepa.py` | Nuovi campi `conditioning`, `demo_*`, `language_dropout` (vedi sotto) |
| `src/lerobot/policies/vla_jepa/demo_bank.py` (nuovo) | `DemoVideoBank`: indicizza `<root>/<task_folder>/*.mp4`, sceglie la demo dalla **stringa del task** e campiona 16 frame sull'intero video (stile TSN: casuale in training, centrato in eval). Resize "squash" per non tagliare i contenitori ai lati. Cache in RAM |
| `src/lerobot/policies/vla_jepa/demo_resampler.py` (nuovo) | `DemoResampler` (Perceiver). Il gain del LayerNorm finale è inizializzato all'RMS degli embedding di Qwen, così i token demo partono sulla stessa scala dei token di testo |
| `src/lerobot/policies/vla_jepa/qwen_interface.py` | Token `<\|demo\|>` aggiunto **dopo** i token esistenti (ID invariati, nessun resize degli embedding). Nuovo prompt con la demo; per C passa i frame al processor con `do_sample_frames=False` e metadati fps |
| `src/lerobot/policies/vla_jepa/modeling_vla_jepa.py` | `_encode_demos` (V-JEPA2 → resampler, oppure frame per Qwen); hook su `embed_tokens` per l'iniezione; demo nel `forward` e nel `predict_action`; nella policy: bank, language dropout, **demo fissata per episodio** (scelta a `reset()`) |
| `run_train_scripts/demo_task_maps/ur5e_pick_place_human.json` (nuovo) | istruzione → `task_XX` (id = 4·colore + contenitore; colori green, yellow, blue, red; contenitori first…fourth. Verificato visivamente sui video) |
| `run_train_scripts/train_ur5e_vla_jepa_demo.sh` (nuovo) | Script SLURM auto-rilanciante, come `train_ur5e_vla_jepa.sh` |

Con `conditioning=language` (default) il comportamento è **identico** all'originale: stesso prompt,
nessun token nuovo, nessun modulo nuovo. I checkpoint esistenti restano compatibili.

### Parametri di config

| Campo | Default | Significato |
|---|---|---|
| `conditioning` | `language` | `language` (originale), `demo` (solo demo), `both` (demo + istruzione) |
| `demo_encoder` | `vjepa` | `vjepa` (A) oppure `qwen_video` (C) |
| `demo_root` | – | cartella con `task_XX/video_NNN.mp4` |
| `demo_task_map_path` | – | JSON istruzione → cartella |
| `demo_num_frames` | 16 | frame campionati sull'intera demo |
| `demo_image_size` | 256 (A) / 224 (C) | lato del resize quadrato |
| `demo_num_eval_per_task` | 8 | ultimi N video di ogni task riservati all'eval (training usa i primi 32 di 40) |
| `demo_num_tokens` | 64 | token `<\|demo\|>` (solo A) |
| `demo_resampler_depth` / `_num_heads` | 2 / 16 | dimensioni del resampler |
| `language_dropout` | 0.0 | con `both`: probabilità di togliere l'istruzione in training |
| `demo_cache_videos` | true | cache dei video decodificati: circa 3.6 GB di RAM per rank (A) con 12 task × 32 video |

**Perché si usa la stringa e non `task_index`:** gli split con task rimossi rinumerano `task_index`
(in `delta_removed_0_5_10_15`, l'indice 0 è "green → second bin"). La stringa invece è stabile.

### Dati

- Robot: dataset LeRobot UR5e esistenti (`open_x_embodiment/datasets/*_lerobot`), senza modifiche.
- Demo umane: `/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset/pick_place/human_rgb_pick_place/video/task_{00..15}/video_{000..039}.mp4`
  (672×376, 10 fps, circa 45–50 frame). Sono le stesse demo `human_rgb` del Multi-Task-LFD framework.
- Con `delta_removed_0_5_10_15` i task 0, 5, 10 e 15 non compaiono mai in training, né come robot
  né come demo. Sono il test **one-shot su task nuovi**.

## 3. Step di addestramento

Tutti i comandi si lanciano da `/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot`.
Lo script attiva da solo l'env `lerobot` e si rilancia allo scadere delle 7 h di `gpuq`.
Default: 4×GPU, batch 4 per GPU (16 effettivo), `NUM_EPOCHS=10`, **checkpoint alla fine di ogni epoca**
(`SAVE_FREQ` = ceil(total_frames / 16), `STEPS` = `NUM_EPOCHS`·`SAVE_FREQ`, decay dell'LR allineato a `STEPS`),
warm start da `lerobot/VLA-JEPA-Pretrain`. Esempio: `delta_all` ha 109143 frame → 6822 step per epoca → 68220 step.
Il dataset si sceglie con `DATASET_NAME` (default `ur5e_pick_place_delta_removed_0_5_10_15`).

### Step 0: smoke test (già eseguito, 2026-09-30, job 691285)

`outputs/smoke_vla_jepa_demo/smoke.sh`: 4 step di training su 1 GPU con batch 2, poi `select_action`
dal checkpoint salvato, per A e per C. Entrambi `TRAIN_EXIT 0` / `EVAL CHECK OK`, circa 1.4 s/step.
A inferenza la demo cambia da un episodio all'altro, come previsto.

### Step 1: baseline a linguaggio (esiste già)

`outputs/ur5e_vla_jepa_ur5e_pick_place_delta_removed_0_5_10_15/checkpoints/032000`: stesso dataset,
stesso numero di step. Serve come riferimento per le curve (`action_loss`, `wm_loss`) e per il success rate.

### Step 2: opzione A, solo demo (esperimento principale)

```bash
sbatch run_train_scripts/train_ur5e_vla_jepa_demo.sh
# = CONDITIONING=demo DEMO_ENCODER=vjepa DATASET_NAME=ur5e_pick_place_delta_removed_0_5_10_15
# output: outputs/ur5e_vla_jepa_demo_vjepa_ur5e_pick_place_delta_removed_0_5_10_15
```

### Step 3: opzione C, solo demo (baseline)

```bash
DEMO_ENCODER=qwen_video sbatch run_train_scripts/train_ur5e_vla_jepa_demo.sh
```

Il confronto A vs C, a parità di dati e di step, isola l'effetto dello spazio V-JEPA rispetto allo
spazio semantico di Qwen.

### Step 4 (opzionale): curriculum linguaggio → demo

Il checkpoint preaddestrato si aspetta "Your task is {instruction}". Se A parte lento (action_loss
molto più alta della baseline nei primi migliaia di step), si può passare dal linguaggio alla demo in modo graduale:

```bash
# 4a: demo + istruzione, istruzione tolta nel 50% dei campioni
CONDITIONING=both LANGUAGE_DROPOUT=0.5 NUM_EPOCHS=5 sbatch run_train_scripts/train_ur5e_vla_jepa_demo.sh
# 4b: solo demo, warm start da 4a
PRETRAINED_PATH=$PWD/outputs/ur5e_vla_jepa_both_vjepa_ur5e_pick_place_delta_removed_0_5_10_15/checkpoints/<ultima epoca>/pretrained_model \
CONDITIONING=demo NUM_EPOCHS=5 JOB_NAME=ur5e_vla_jepa_demo_vjepa_curriculum \
  sbatch run_train_scripts/train_ur5e_vla_jepa_demo.sh
```

Anche un modello `both` può essere valutato senza istruzione: a inferenza manda un task, ma
`language_dropout` agisce solo in training. Per l'eval "solo demo" basta impostare
`"conditioning": "demo"` nel `config.json` del checkpoint.

### Step 4b: loss contrastiva demo–testo (2026-10-01, job 777952)

Motivazione: il t-SNE (`analysis/demo_tsne/`) mostra che, con sola L_FM + β·L_WM, il Perceiver codifica il
contenitore (5-NN 0.99) ma scarta il colore della scatola (0.29–0.38 contro 0.80 di V-JEPA2, caso 0.25), anche sulle
demo di training. Si aggiunge una InfoNCE fra la demo e il testo del proprio task:

    d̄  = g_ω(mean_k d_k)                      (testa MLP 2048 → 2048 → 4096)
    t_c = [cn(mean_j h^(20)_j(ℓ_c)), cn(h^(28)_last(ℓ_c))]   (Qwen base congelato, calcolate una volta)
    L_con = −log softmax_c'( cos(d̄, t_c') / τ )_c ,  c' ∈ task di training,  τ appresa (init 0.07)
    L = L_FM + β·L_WM + λ·L_con,  λ = 0.1

`cn` = centrata sui task e normalizzata. Le ancore grezze (media dell'ultimo layer) hanno coseno ≥ 0.998 fra task
diversi e non discriminano: la media degli hidden state codifica soprattutto il colore, l'ultimo token il
contenitore, quindi si concatenano (stesso contenitore +0.35, stesso colore +0.30, entrambi diversi −0.32).

Metriche su wandb: `demo_text_acc`, `demo_text_color_acc`, `demo_text_bin_acc`, `demo_text_margin`,
`demo_text_pos_sim`, `demo_text_temperature`.

```bash
CONTRASTIVE_WEIGHT=0.1 DATASET_NAME=ur5e_pick_place_delta_removed_0_5_10_15 \
  sbatch run_train_scripts/train_ur5e_vla_jepa_demo.sh
# output: outputs/ur5e_vla_jepa_demo_vjepa_con0.1_ur5e_pick_place_delta_removed_0_5_10_15
```

### Step 5: sim → real

Come in `train_ur5e_vla_jepa_real.sh`: warm start dal checkpoint sim finale, dataset reale. Le demo
umane sono già reali.

```bash
PRETRAINED_PATH=$PWD/outputs/ur5e_vla_jepa_demo_vjepa_ur5e_pick_place_delta_removed_0_5_10_15/checkpoints/<epoca migliore>/pretrained_model \
DATASET_NAME=real_ur5e_pick_place_delta_removed_0_5_10_15 NUM_EPOCHS=10 \
JOB_NAME=ur5e_vla_jepa_demo_vjepa_real sbatch run_train_scripts/train_ur5e_vla_jepa_demo.sh
```

### Step 6 (futuro, non implementato): stage 1 video-conditioned

Per un modello "video-conditioned" anche nel pretraining (Sez. 3.2, SSv2), bisogna rifare lo stage 1
con demo al posto di ℓ. SSv2 ha 174 classi-template: si accoppia ogni clip target con un'altra clip
della **stessa classe**, con oggetti e scene diverse, usata come demo. Questo passo richiede un loader
SSv2 (solo `wm_loss`, niente azioni) e molte più risorse. Ha senso solo se gli step 2–4 mostrano che il
condizionamento via demo funziona ma è limitato dal pretraining a linguaggio.

## 4. Diagnostica durante il training

- **`wm_loss` rispetto alla baseline**: deve restare dello stesso ordine. Se crolla molto sotto la
  baseline, c'è un leakage da cercare (non dovrebbe succedere con demo umane di altri episodi).
- **`action_loss`**: nei primi step A parte più alta di C, perché il resampler è inizializzato a caso.
  Se non converge verso la baseline a linguaggio, prova lo step 4.
- **La demo viene ignorata?** Il frame I_t0 da solo non dice quale cubo o quale contenitore scegliere,
  quindi la demo è necessaria. Si verifica all'eval con lo **scambio della demo**: copia la cartella
  `pretrained_model` e punta `demo_task_map_path` nel `config.json` a una mappa permutata (ad esempio
  "green → first" → `task_05`). Il robot deve eseguire il task *della demo*, non quello del prompt.

## 5. Valutazione (VLA-Bench, robosuite)

Il client non cambia. Il server (`run_eval_scripts/lerobot_policy_server.py`) passa già
`batch["task"]` e chiama `policy.reset()` a ogni episodio. La policy allora sceglie una demo **tra
quelle di eval** (le ultime 8 per task, a rotazione) e la tiene per tutto l'episodio.

```bash
cd /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/VLA-Benchmark/robosuite_test
CKPT=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/outputs/ur5e_vla_jepa_demo_vjepa_ur5e_pick_place_delta_removed_0_5_10_15/checkpoints/<epoca>/pretrained_model
# Usage: run_lerobot_eval.sh [config_yml] [policy_path] [port] [run_number] [num_trials_per_task] [task_suite_name] [otd] [object_set]
sbatch run_lerobot_eval.sh models/lerobot_vla_jepa_eval_config.yml $CKPT 8770 0 10 pick_place false   # task di training
sbatch run_lerobot_eval.sh models/lerobot_vla_jepa_eval_config.yml $CKPT 8771 0 30 pick_place true    # task held-out 0/5/10/15
```

Tabella da compilare (success rate):

| Modello | Task visti | Task held-out (one-shot) |
|---|---|---|
| Linguaggio (step 1) | | |
| A: V-JEPA demo (step 2) | | |
| C: Qwen video demo (step 3) | | |
| A + curriculum (step 4) | | |

## 6. Limiti noti

- A inferenza la demo viene ricodificata a ogni chunk. Il costo è piccolo rispetto al LLM, ma si
  potrebbe mettere in cache l'output del resampler o il prefisso in KV-cache.
- Il resampler parte da inizializzazione casuale e il LLM deve imparare a leggere lo spazio V-JEPA
  durante lo stage 2 (vedi step 4).
- La cache delle demo sta nel processo principale di ogni rank. Il primo passaggio sulle demo è più
  lento (decodifica mp4, circa 0.3–0.4 s a video), poi è trascurabile. Per disattivarla: `--policy.demo_cache_videos=false`.
