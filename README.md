# Neuralyzer — Multimodal Dementia Detection

Faithful reimplementation of **"A Multimodal Framework for Dementia Detection via
Linguistic and Acoustic Representation Learning."** Combines a HuBERT speech
branch and a BERT text branch, aligns them via Mutual Information (MINE), fuses
them with attention (AT-Fusion), and classifies AD vs. control.

> **Workflow:** heavy training runs on **Google Colab (GPU)** and produces a
> single checkpoint `best_model.pt`. You download that file and run **inference
> locally** — no training dependencies needed on your machine.

---

## Architecture

```
 Audio ─▶ HuBERT (sum last 2 layers) ─▶ ASP pooling ─▶ avg chunks ─▶ Linear ─▶ f_a ┐
                                                                                   ├─▶ MINE  (maximize I(f_a; f_t))
 Text  ─▶ BERT  ([CLS] embedding) ───────────────────────────────────────▶ f_t ───┘
                                                                                   │
                                            (f_a, f_t) ─▶ AT-Fusion ─▶ h ─▶ Classifier ─▶ {control, AD}

 Loss = CrossEntropy  +  λ · (−MINE)     (λ = 0.25)
```

## Project structure

```
src/
  configs/      # dataclass config + default.yaml (all knobs; nothing hardcoded)
  utils/        # seeding, logging, device
  datasets/     # base contract, collator (10s chunking + tokenize), mock, process, kaggle_pitt, adress*, process2*, factory
  modules/      # pooling (ASP/mean/max), fusion (AT/concat/GMU/MUTAN/MFB/MFH/BLOCK), MINE, classifier
  models/       # speech_encoder, text_encoder, multimodal_model
  losses/       # MINE DV bound + combined loss
  trainer/      # training loop, early stopping, StepLR, checkpointing
  evaluation/   # metrics (acc/prec/recall/spec/F1), ablation grids
train.py        # multi-seed training  (run on Colab) → runs/best_model.pt
ablate.py       # ablation studies      (run on Colab)
inference.py    # local prediction from best_model.pt
scripts/smoke_test.py        # end-to-end mock pipeline check
scripts/transcribe_whisper.py   # Whisper transcripts for audio-only datasets (kaggle_pitt)
notebooks/train_on_colab.ipynb   # Colab: install → smoke → train → download .pt
requirements-train.txt      # Colab (training) deps
requirements-inference.txt  # local (inference-only) deps
wheels/         # cached torch wheel for offline local install
```
`*` ADReSS / PROCESS-2 loaders are placeholders until the gated data is obtained.

---

## A. Train on Colab

1. Push this repo to GitHub (see "Push to GitHub" below).
2. Open `notebooks/train_on_colab.ipynb` in Colab, set **Runtime → GPU**.
3. Run the cells: clone → install → smoke test → train → download `best_model.pt`.

Smoke / training commands (also runnable in a Colab cell):
```bash
python scripts/smoke_test.py
# mock (pipeline check, meaningless metrics):
python train.py --config src/configs/default.yaml --dataset mock --single-seed --max-epochs 3
# real open data (PROCESS Challenge, English speech):
python train.py --config src/configs/default.yaml --dataset process --data-root /content/pitt
```

### Open dataset: PROCESS Challenge (`dataset: process`)

Real, English, no data-use-agreement (only a free Kaggle login):
[`tahouramorovati/dementia-detection-using-speech`](https://www.kaggle.com/datasets/tahouramorovati/dementia-detection-using-speech)
— a **PROCESS Challenge** re-upload: 157 labeled `Process-rec-XXX` participants,
each with 3 task recordings (**CTD** Cookie Theft, **PFT** phonemic fluency,
**SFT** semantic fluency) plus co-located `.txt` transcripts, and labels in
`Data_AUG_13.11.2024_output.csv` (`Record-ID`, `Class` = HC/MCI/Dementia).

The loader uses the **CTD** task by default (matches the paper's picture
description), reads the co-located transcript (no Whisper needed), and maps the
label **HC → 0, MCI + Dementia → 1** (the paper's binary grouping):
```bash
# 1) download (needs kaggle.json token) and unzip
kaggle datasets download -d tahouramorovati/dementia-detection-using-speech -p /content/pitt --unzip
# 2) train (auto-detects the labels CSV under the data root)
python train.py --config src/configs/default.yaml --dataset process --data-root /content/pitt
# use all three tasks (3x samples) instead of just CTD:
python train.py --dataset process --data-root /content/pitt --tasks CTD,SFT,PFT
```
Override the class mapping or CSV if needed:
```yaml
data:
  label_map: {HC: 0, MCI: 1, Dementia: 1}
  labels_csv: /content/pitt/Data_AUG_13.11.2024_output.csv
  process_tasks: [CTD]
```
> **License:** derived from PROCESS/DementiaBank data. Personal reproduction only.
>
> The same Kaggle upload also bundles Pitt transcripts (`Training_No.csv`) with
> no audio, so only the PROCESS records are usable for the multimodal model.

## B. Run inference locally

> The full pipeline (HuBERT + BERT → ASP → MINE + AT-Fusion → classifier) is
> **verified to run locally on CPU** via `scripts/smoke_test.py` with real
> pretrained weights (correct shapes, decreasing loss).

Create the venv **outside the repo** (this workspace periodically runs
`git clean`, which deletes an in-repo `.venv/` since it's git-ignored), then
install the lightweight inference deps:
```powershell
py -3.13 -m venv C:\Users\lukra\neuralyzer_venv
C:\Users\lukra\neuralyzer_venv\Scripts\Activate.ps1
pip install wheels\torch-2.9.0+cpu-cp313-cp313-win_amd64.whl   # cached torch, avoids re-download
pip install -r requirements-inference.txt
```

Then predict with the checkpoint you downloaded from Colab:
```powershell
python inference.py --checkpoint best_model.pt --audio recording.wav --transcript "the boy is taking a cookie ..."
# or:
python inference.py --checkpoint best_model.pt --audio recording.wav --transcript-file recording.txt
```

### Local troubleshooting (flaky-network workarounds, verified)
- **In-repo `.venv/` vanishes:** the workspace runs `git clean` on ignored files. Keep the venv **outside the repo** (as above).
- **PyPI (`files.pythonhosted.org`) resets after ~1 MB:** push installs through with heavy retries so partial downloads accumulate — `pip install ... --retries 300 --resume-retries 2000`. PyTorch's CDN and HuggingFace are unaffected.
- **HuggingFace model download crashes** (`error decoding response body` from the Xet backend): set `HF_HUB_DISABLE_XET=1` before downloading models (falls back to the classic resumable path).
- **transformers version:** pinned `>=4.40,<5` — `AutoModel`/HuBERT API is stable there; 5.x risks API breaks (verified working on `transformers==4.57.1`).

---

## Configuration

Everything is in `src/configs/default.yaml`. Key settings:

| Setting | Value | Source |
| --- | --- | --- |
| Speech encoder | `facebook/hubert-base-ls960`, sum last 2 layers | paper |
| Pooling | Attentive Statistics Pooling | paper |
| Text encoder | `bert-base-uncased`, `[CLS]` | paper |
| Fusion | AT-Fusion | paper |
| Loss | CE + 0.25·MINE | paper |
| Batch size | 8 | paper |
| Val split | 65/35 stratified | paper |
| Optimizer | AdamW, wd 0.01 | **user decision** |
| Learning rate | **1e-5** (peak) | **user-optimized** (paper-era ~2e-5) |
| Scheduler | **linear warmup + decay** (10% warmup) | **user-optimized** (paper: StepLR 4/0.1) |
| Max epochs / patience | **250 / 20** | **user-optimized** (paper: 100 / 8) |
| MINE net | 2 hidden Linear+ReLU (512) | **user decision** |
| Classifier | 768→128→2, dropout 0.1 | **user decision** |
| Seeds | 42–46 (5 runs, mean±std) | **user decision** |

> **Paper-exact training:** set `train.scheduler: steplr`, `lr: 2.0e-5`,
> `max_epochs: 100`, `early_stopping_patience: 8` in the YAML (or pass
> `--scheduler steplr --lr 2e-5 --max-epochs 100 --patience 8`).
>
> **CLI overrides** (train.py): `--dataset --data-root --transcripts-file
> --label-map '{"Dementia":1,"Control":0}' --labels-csv --tasks --lr
> --batch-size --max-chunks --max-epochs --patience --scheduler --output-dir
> --single-seed`.

### GPU memory / CUDA OOM

The speech encoder runs HuBERT over **every 10s chunk of the whole batch at
once**, so long recordings (e.g. PROCESS) can blow past a ~15 GB GPU. Defaults
are tuned to fit a free Colab **T4** at batch size 8 (kept at 8 because MINE
needs a real batch to estimate mutual information):

- `use_amp: true` — fp16, ~half the activation memory.
- `gradient_checkpointing: true` — recompute activations in backward.
- `max_chunks_per_recording: 4` — cap at 40s of audio per recording.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — set automatically by
  `train.py`; reclaims "reserved but unallocated" memory (fixes fragmentation OOM
  in `.backward()`).

Still OOM? Turn these levers (cheapest first):
```bash
python train.py --dataset process --data-root /content/pitt --max-chunks 3   # 30s cap
python train.py --dataset process --data-root /content/pitt --batch-size 4   # last resort (weakens MINE)
```

### Multi-GPU (e.g. Kaggle 2xT4)

A single `train.py` process uses **one** GPU. To use several GPUs, run the
seed-parallel launcher: it starts one `train.py` process per GPU (pinned via
`CUDA_VISIBLE_DEVICES`), splits the seeds across them, streams both logs live,
and aggregates every `gpuN/summary.csv` into one mean +/- std table at the end.
MINE's per-run batch stays intact (we parallelize *seeds*, not the model).

```bash
python train_parallel.py --dataset process \
  --data-root /kaggle/input/<folder> \
  --output-dir /kaggle/working/outputs \
  --seeds 42,43,44,45,46            # round-robined across all visible GPUs
# add --gpus 0,1 to restrict devices; extra flags (--max-chunks, --batch-size, ...) are forwarded
```

Best checkpoints land at `<output-dir>/gpuN/best_model.pt`.

## Ablations (paper Sec VII)

```bash
python ablate.py --study pooling   # ASP vs mean vs max
python ablate.py --study ssl       # HuBERT vs wav2vec2 vs XLS-R
python ablate.py --study lambda    # 0 / 0.1 / 0.2 / 0.25 / 0.3
python ablate.py --study fusion    # concat / GMU / MUTAN / MFB / MFH / BLOCK / AT-Fusion
python ablate.py --study layers    # last 1 / 2 / 3 HuBERT layers
```

## Datasets

- **Mock** (`dataset: mock`): synthetic, for pipeline validation only — metrics are meaningless.
- **PROCESS** (`dataset: process`): open English speech (free Kaggle login); CTD task, transcripts included, HC=0 / MCI+Dementia=1. See the workflow above. License caveat applies.
- **Kaggle Pitt** (`dataset: kaggle_pitt`): folder-labeled + Whisper-transcript loader — kept for other audio-only Kaggle uploads; not the dataset above.
- **ADReSS / PROCESS-2**: gated (DementiaBank / HuggingFace). Loaders are stubs
  raising `NotImplementedError` until the real data and its exact format can be
  inspected — per the reproduction rule, unspecified details are not guessed.

## Caveats

- Mock metrics carry **no scientific meaning**; only real ADReSS/PROCESS-2 results do.
- Some hyperparameters (optimizer, MINE net, bilinear-fusion factors) were
  unspecified in the paper; the chosen defaults are documented above and in
  `src/configs/config.py`.

## Push to GitHub

```powershell
git add -A
git commit -m "Clean reimplementation: Colab-train / local-inference"
git branch -M main
git remote add origin https://github.com/<your-username>/Neuralyzer.git
git push -u origin main
```
