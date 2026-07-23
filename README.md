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
  datasets/     # base contract, collator (10s chunking + tokenize), mock, adress*, process2*, factory
  modules/      # pooling (ASP/mean/max), fusion (AT/concat/GMU/MUTAN/MFB/MFH/BLOCK), MINE, classifier
  models/       # speech_encoder, text_encoder, multimodal_model
  losses/       # MINE DV bound + combined loss
  trainer/      # training loop, early stopping, StepLR, checkpointing
  evaluation/   # metrics (acc/prec/recall/spec/F1), ablation grids
train.py        # multi-seed training  (run on Colab) → runs/best_model.pt
ablate.py       # ablation studies      (run on Colab)
inference.py    # local prediction from best_model.pt
scripts/smoke_test.py   # end-to-end mock pipeline check
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

Smoke / real-data commands (also runnable in a Colab cell):
```bash
python scripts/smoke_test.py
python train.py --config src/configs/default.yaml --dataset mock --single-seed --max-epochs 3
# real data (after implementing the loader):
python train.py --config src/configs/default.yaml --dataset adress --data-root /content/adress
```

## B. Run inference locally

Install the lightweight inference deps. If the network is flaky, install the
cached torch wheel first:
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install wheels\torch-2.9.0+cpu-cp313-cp313-win_amd64.whl   # optional, avoids re-downloading torch
pip install -r requirements-inference.txt
```

Then predict with the checkpoint you downloaded from Colab:
```powershell
python inference.py --checkpoint best_model.pt --audio recording.wav --transcript "the boy is taking a cookie ..."
# or:
python inference.py --checkpoint best_model.pt --audio recording.wav --transcript-file recording.txt
```

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
| Batch size / patience | 8 / 8 | paper |
| Scheduler | StepLR (step 4, γ 0.1) | paper |
| Val split | 65/35 stratified | paper |
| Optimizer | AdamW, lr 2e-5, wd 0.01 | **user decision** |
| MINE net | 2 hidden Linear+ReLU (512) | **user decision** |
| Classifier | 768→128→2, dropout 0.1 | **user decision** |
| Seeds | 42–46 (5 runs, mean±std) | **user decision** |

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
