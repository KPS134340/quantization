# Phase 2 — Mechanistic Decomposition
## Complete Run Guide

---

## What Phase 2 Does

Phase 1 established *that* quantization changes moral outputs.
Phase 2 establishes *where* and *why* — specifically that failures
cluster around temporally extended consequentialist scenarios and that
the mechanism is degradation of sequential reasoning chains.

---

## File Overview

```
phase2/
├── 01_build_strata.py        — Selects 400 scenarios across 4 strata
├── 02_run_cot_inference.py   — Runs 5-step CoT prompts through all models
├── 03_classify_frameworks.py — Llama 3.3 70B classifies frameworks + coherence
├── 04_failure_taxonomy.py    — Classifies each FP16 vs INT4 pair (Types 0–4)
├── 05_statistical_tests.py   — Four statistical tests + two key figures
├── 06_qualitative_cases.py   — Selects 30 cases for the paper
└── README.md                 — This file

data/phase2/
├── stratified_400.csv        — The 400-scenario stratified sample
├── cot_outputs_*.csv         — Full CoT outputs (large files)
├── annotations_*.csv         — LLM judge framework + coherence labels
├── failure_pairs_*.csv       — FP16 vs INT4/INT8 paired comparison
├── failure_summary_*.csv     — Failure type counts per stratum
├── token_flip_log_*.csv      — First failure step per Stratum A item
├── stats_primary_*.csv       — Logistic regression results
├── stats_secondary_*.csv     — Execution vs framework failure counts
├── stats_tertiary_*.csv      — Chain length fingerprint data
├── stats_quaternary_*.csv    — INT8 vs INT4 gradient data
├── qualitative_cases_*.csv   — 30 selected cases
└── qualitative_display_*.txt — Human-readable case formatting

figures/
├── phase2_chain_length_curve_*.png   — The central empirical figure
└── phase2_degradation_gradient_*.png — INT8 vs INT4 comparison
```

---

## Prerequisites

Same hardware and libraries as Phase 1, plus:

```bash
pip install openai
```

### LLM Judge Setup (Free — No Cost)

Phase 2 uses **Llama 3.3 70B** as an LLM judge for framework classification
and chain coherence scoring. This runs via **Groq's free API tier** — no
payment required.

**Option 1 — Groq free tier (RECOMMENDED):**

1. Create a free account at https://console.groq.com
2. Generate an API key at https://console.groq.com/keys
3. Set the environment variable:

```bash
export GROQ_API_KEY="gsk_..."
```

> **Why Llama 3.3 70B instead of GPT-4o?**
> - **Free:** Groq's free tier provides zero-cost inference for Llama 3.3 70B
> - **Reproducible:** Open-weight model makes the entire pipeline reproducible
> - **Capable:** Llama 3.3 70B matches GPT-4o on classification benchmarks
> - **Methodologically sound:** Using an open model as judge avoids dependency
>   on a proprietary system whose behavior may change between API versions

**Option 2 — Local vLLM server (for self-hosted deployments):**

```bash
# Start the vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct --port 8000

# Tell the classifier to use local server
export JUDGE_PROVIDER="local"
```

**Groq free tier rate limits:** Approximately 30 requests per minute for
70B models. Script 3 handles rate limiting automatically with exponential
backoff. Total annotation time is ~4–6 hours due to rate limits (vs 2–4 hours
with a paid API), but the cost is $0.

Phase 1 must be fully complete before starting Phase 2.
Specifically, data/raw/all_datasets_combined.csv must exist.

---

## Step-by-Step Execution

### Step 1 — Build the stratified 400-scenario sample

```bash
python 01_build_strata.py
```

**What it does:** Reads the Phase 1 datasets and selects 400 scenarios
distributed across the four strata. Uses keyword filters to ensure
Stratum A items genuinely contain temporal/causal language.

**Expected output:**
```
Stratum A: selected 150/150 items (pool had 2800)
Stratum B: selected 100/100 items (pool had 1200)
Stratum C: selected 100/100 items (pool had 1800)
Stratum D: selected 50/50  items (pool had 900)
Total: 400 scenarios across 4 strata
```

**Runtime:** Under 1 minute.

---

### Step 2 — Run structured CoT inference

```bash
python 02_run_cot_inference.py
```

**What it does:** Runs all 400 scenarios through both models at all
three precision levels using the 5-step structured CoT prompt. Saves
the full reasoning text for every run.

**Expected runtime:** ~12–16 hours on an A100.
This is longer than Phase 1 because we generate 600 tokens per item
instead of 5. Leave it running overnight.

Saves progress after each precision condition — if it crashes, re-run
the same command to resume without losing completed work.

**To test first with a subset:**
Edit the last line of the file to:
```python
run_phase2_inference(model_keys=["mistral"], precisions=["FP16", "INT4"])
```
This takes ~5 hours and confirms the pipeline works.

**Expected output files:**
- data/phase2/cot_outputs_llama3.csv   (~300 MB)
- data/phase2/cot_outputs_mistral.csv  (~300 MB)

---

### Step 3 — Classify frameworks and score chain coherence

```bash
export GROQ_API_KEY="gsk_..."   # set your free Groq key
python 03_classify_frameworks.py
```

**What it does:** For every CoT output row, calls Llama 3.3 70B (via Groq) twice:
  (a) To classify which ethical framework the model used at Step 3
  (b) To score each of the four step-to-step transitions (1→2, 2→3, 3→4, 4→5)
      as coherent (1) or incoherent (0)

Also creates the 15% manual verification sample CSV.

**LLM Judge:** Llama 3.3 70B via Groq free tier (model: `llama-3.3-70b-versatile`)

**Expected runtime:** 4–6 hours (Groq free tier rate limits are the bottleneck).
Saves progress every 50 rows — safe to interrupt and resume.

**Cost: $0** (Groq free tier)

**After this step completes:**
Open data/phase2/manual_verification_llama3.csv (and mistral).
Fill in the human_framework and human_coherence_* columns for each row.
These are the rows a human annotator reviews. Target: κ ≥ 0.75.
Once filled, run:
```python
from classify_frameworks import compute_interrater_kappa
compute_interrater_kappa("data/phase2/manual_verification_llama3.csv")
```

---

### Step 4 — Classify failure types

```bash
python 04_failure_taxonomy.py
```

**What it does:** Joins FP16 and INT4 annotations by scenario ID.
For each pair, applies the failure type decision logic:

  Type 0 — No failure
  Type 1 — Conclusion drift (framework matches, conclusion doesn't)
  Type 2 — Framework switch (different framework applied)
  Type 3 — Chain collapse (coherence drops sharply mid-chain)
  Type 4 — Full collapse (incoherent output)

Also builds the token-flip event log for Stage 3.

**Runtime:** Under 1 minute.

**Key output to check:**
```
Failure rate by stratum (FP16 vs INT4):
stratum  n_failures  total  overall_failure_rate
A           ???       150       ???
B           ???       100       ???
C           ???       100       ???
D           ???        50       ???
```
The hypothesis predicts Stratum A failure rate >> Stratum B and C.

---

### Step 5 — Run statistical tests

```bash
python 05_statistical_tests.py
```

**What it does:** Runs four tests and generates two key figures.

PRIMARY: Logistic regression confirming Stratum A has elevated odds
  ratio relative to Stratum B. Look for odds_ratio(A) significantly > 1.

SECONDARY: Within Stratum A failures, checks whether execution failures
  (Type 1+3) dominate over framework switch failures (Type 2).
  If yes: mechanism is procedural degradation, not misidentification.

TERTIARY: Plots the chain length fingerprint — the central figure of
  the paper. X = FP16 chain length, Y = P(failure under INT4).
  If monotonically increasing: this single figure tells the whole story.

QUATERNARY: Compares INT8 vs INT4 curves. If INT4 is steeper, aggressive
  compression specifically damages long-chain reasoning.

**Runtime:** Under 2 minutes.

**Output figures:**
- figures/phase2_chain_length_curve_*.png    ← Paper Figure 2
- figures/phase2_degradation_gradient_*.png  ← Paper Figure 3

---

### Step 6 — Select qualitative cases

```bash
python 06_qualitative_cases.py
```

**What it does:** Selects the 30 best Type 1 and Type 3 failure examples
from Stratum A. Prioritizes cases where:
  - FP16 was highly coherent (score ≥ 0.75)
  - INT4 clearly broke down (score ≤ 0.50)
  - The exact failure transition is identifiable

Writes a human-readable formatted text file for easy inclusion in the
paper's qualitative evidence section.

**Runtime:** Under 1 minute.

---

## Complete Runtime Estimate

| Step | Script | Runtime | Cost |
|------|--------|---------|------|
| 1 | 01_build_strata.py | < 1 min | $0 |
| 2 | 02_run_cot_inference.py | 12–16 hours | $0 (GPU) |
| 3 | 03_classify_frameworks.py | 4–6 hours | **$0** (Groq free) |
| 4 | 04_failure_taxonomy.py | < 1 min | $0 |
| 5 | 05_statistical_tests.py | < 2 min | $0 |
| 6 | 06_qualitative_cases.py | < 1 min | $0 |
| **Total** | | **~18–22 hours** | **$0** |

Step 2 dominates (GPU inference). Step 3 is slower than a paid API due
to free tier rate limits, but costs nothing. Steps 1 and 4–6 together
take under 5 minutes.

---

## LLM Judge Details

| Property | Value |
|---|---|
| **Model** | Llama 3.3 70B Instruct |
| **Provider** | Groq (free tier) |
| **Model ID** | `llama-3.3-70b-versatile` |
| **API format** | OpenAI-compatible |
| **Cost** | $0 |
| **Rate limits** | ~30 RPM (handled automatically) |
| **Alternative** | Local vLLM server (`JUDGE_PROVIDER=local`) |

**Why this model?** Llama 3.3 70B is the strongest open-weight model
available for free inference. It matches GPT-4o-class performance on
classification and structured reasoning tasks. Using an open model also
makes the entire pipeline fully reproducible without proprietary API
dependencies.

---

## What Feeds Into Later Stages

| Output File | Used By |
|---|---|
| data/phase2/token_flip_log_*.csv | Stage 3 (CRWP) — identifies which weights to protect |
| data/phase2/stratified_400.csv | Stage 2 (moral manifold construction) |
| data/phase2/qualitative_cases_*.csv | Paper Section 3.2 |
| figures/phase2_chain_length_curve_*.png | Paper Figure 2 |
| data/phase2/stats_primary_*.csv | Paper Table 1 |

---

## How to Download Results

From a cloud instance (e.g. RunPod), on your LOCAL machine:

```bash
scp -r <your-instance-ip>:~/phase2/data/phase2  ./phase2_data
scp -r <your-instance-ip>:~/phase2/figures       ./phase2_figures
```

Or use JupyterLab's file browser to download individual files.
