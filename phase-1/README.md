# Phase 1 — Output Divergence Audit
## Complete Run Guide

---

## What this code does

Runs every dataset item through Llama-3-8B and Mistral-7B at three 
precision levels (FP16, INT8, INT4), records each model's answer, 
then measures how often the compressed versions disagree with FP16.

---

## File Overview

```
phase1/
├── 00_setup.py          — checks your environment before anything runs
├── 01_load_datasets.py  — downloads all 4 datasets, saves to data/raw/
├── 02_load_models.py    — loads models at FP16 / INT8 / INT4 (smoke test)
├── 03_run_inference.py  — main loop: runs every item through every model
├── 04_compute_metrics.py— agreement rates + Cohen's kappa
├── 05_visualize.py      — generates the heatmap figures
└── README.md            — this file

data/
├── raw/                 — CSVs from step 1
├── results/             — inference outputs from step 3
└── metrics/             — computed metrics from step 4

figures/                 — PNG heatmaps from step 5
```

---

## Prerequisites

### 1. Get a cloud GPU

You need at least 24 GB VRAM. Cheapest options:

- **RunPod**      → runpod.io  (A100 40GB ~$1.50/hr, easiest setup)
- **Lambda Labs** → lambdalabs.com (similar pricing)
- **Google Colab Pro+** → colab.google (T4/A100, pay per usage)

On RunPod: create an account → New Pod → select A100 40GB → 
select PyTorch 2.x template → Start → Connect via SSH or JupyterLab.

### 2. Install dependencies

Once inside your cloud instance terminal:

```bash
pip install transformers bitsandbytes datasets accelerate \
            pandas numpy scikit-learn seaborn matplotlib tqdm torch
```

### 3. Log in to HuggingFace (required for Llama-3)

```bash
pip install huggingface_hub
huggingface-cli login
# Paste your HuggingFace access token when prompted
# Get your token at: https://huggingface.co/settings/tokens
```

Then visit https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
and accept the license agreement (takes a few minutes to activate).

---

## Step-by-Step Execution

Run each script in order from the phase1/ directory.

### Step 0 — Verify environment

```bash
python 00_setup.py
```

What to expect: prints GPU name, VRAM, and library versions.
If anything shows ❌, fix it before proceeding.

---

### Step 1 — Download datasets

```bash
python 01_load_datasets.py
```

What to expect: HuggingFace downloads each dataset (may take 5–15 min 
depending on connection). Saves 4 CSV files to data/raw/.

Prints a summary table like:
```
Dataset      Item Count
ETHICS            9000
MoralBench        2000
MoralStories      2000
ValueBench        1500
Total            14500
```

If a dataset fails to download (HuggingFace IDs sometimes change):
check the HuggingFace Hub for the current identifier and update 
the load_dataset() call in 01_load_datasets.py.

---

### Step 2 — Smoke test model loading

```bash
python 02_load_models.py
```

What to expect: loads Mistral-7B in FP16 → INT8 → INT4, runs one 
forward pass each, unloads between. Should take 5–10 minutes.
Prints the model's answer to a test question at each precision.

If you get a CUDA OOM error on INT4:
→ Restart the instance and try with a larger GPU (80GB A100).
→ Or reduce batch_size in 03_run_inference.py from 8 to 4.

---

### Step 3 — Run inference  ⚠️ This is the long step

```bash
python 03_run_inference.py
```

What to expect: iterates over both models × 3 precisions × all items.
Progress bars show per-batch progress.

Estimated runtime:
- Mistral FP16 : ~45 min
- Mistral INT8 : ~55 min  
- Mistral INT4 : ~60 min
- Llama3 FP16  : ~50 min
- Llama3 INT8  : ~60 min
- Llama3 INT4  : ~65 min
Total          : ~6 hours on A100

Results are saved after each precision run. If it crashes, re-run the 
same command — it detects already-completed runs and skips them.

To test with just one model and two precisions first:
Edit the last line of 03_run_inference.py to:
  run_phase1(model_keys=["mistral"], precisions=["FP16", "INT4"])
This takes ~2 hours and confirms the pipeline works end-to-end.

---

### Step 4 — Compute metrics

```bash
python 04_compute_metrics.py
```

What to expect: runs in under 1 minute. Prints agreement rates and 
kappa scores. Saves CSV files to data/metrics/.

The printed "KEY FINDING" text is the sentence that goes into your 
paper's Phase 1 results section.

---

### Step 5 — Generate figures

```bash
python 05_visualize.py
```

What to expect: runs in under 1 minute. Saves two PNG files per model 
to figures/. These are the heatmaps referenced in the paper.

---

## Downloading Your Results

From a RunPod instance, use scp to copy results to your local machine:

```bash
# Run this on your LOCAL machine terminal (not the cloud instance)
scp -r <your-runpod-ip>:~/phase1/data/metrics ./phase1_metrics
scp -r <your-runpod-ip>:~/phase1/figures      ./phase1_figures
```

Or use JupyterLab's file browser to download individual files.

---

## What the outputs mean

| File | What it tells you |
|---|---|
| data/metrics/llama3_agreement_rates.csv | Per-category % of items where INT4 agreed with FP16 |
| data/metrics/llama3_kappa.csv | Cohen's κ — overall agreement quality |
| data/metrics/llama3_summary.csv | The key finding sentence with exact numbers |
| figures/phase1_divergence_heatmap_*.png | The heatmap figure for the paper |
| figures/phase1_ethics_categories_*.png | ETHICS-only category breakdown |

---

## Interpreting Cohen's kappa

κ > 0.8  → quantization barely changed anything  
κ 0.6–0.8 → noticeable but moderate divergence  
κ 0.4–0.6 → meaningful divergence — worth reporting  
κ < 0.4  → substantial divergence — strong Phase 1 finding  

The hypothesis predicts κ < 0.6 for INT4, especially on 
Utilitarianism and consequence-heavy categories.

---

## Once Phase 1 is complete

The outputs feed directly into Phase 2:
- data/results/ → used to select the 400-scenario stratified sample
- data/metrics/ → divergence rates inform which categories to oversample
- figures/      → go into Section 3.1 of the paper
