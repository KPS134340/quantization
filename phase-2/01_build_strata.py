"""
PHASE 2 — SCRIPT 1
Scenario Stratification: Build the 400-scenario sample.

This script reads the Phase 1 datasets and selects 400 scenarios
distributed across four strata exactly as specified in the methodology:

  Stratum A — Temporally extended consequentialist  (n=150)
  Stratum B — Deontological / rule-based            (n=100)
  Stratum C — Virtue / commonsense                  (n=100)
  Stratum D — Mixed / ambiguous                     (n=50)

The strata are built using dataset splits as proxies for reasoning type
(ETHICS categories map cleanly onto strata). Additional keyword filters
are applied to enforce the selection criteria described in the methodology.

Outputs
-------
  data/phase2/stratified_400.csv   — the 400-scenario sample with stratum labels
  data/phase2/stratum_summary.csv  — counts and source breakdown per stratum
"""

import pandas as pd
import numpy as np
import os
import re

os.makedirs("data/phase2", exist_ok=True)

# ── Random seed for reproducibility ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# ════════════════════════════════════════════════════════════════════════════
# STRATUM DEFINITIONS
# Maps (dataset, category) pairs to stratum labels.
# This is the core design decision — grounded in the methodology.
# ════════════════════════════════════════════════════════════════════════════

# Each entry: (dataset, category) → stratum
STRATUM_MAP = {
    ("ETHICS", "utilitarianism"):  "A",   # Consequentialist reasoning → Stratum A
    ("MoralStories", "acceptability"): "A",  # Action→consequence chains → Stratum A
    ("ETHICS", "deontology"):      "B",   # Rule-based → Stratum B
    ("MoralBench", "general"):     "B",   # Scenario-level, rule-focused → Stratum B
    ("ETHICS", "commonsense"):     "C",   # Social norm judgment → Stratum C
    ("ETHICS", "virtue"):          "C",   # Character evaluation → Stratum C
    ("ValueBench", "unknown"):     "D",   # Framework-ambiguous → Stratum D
}

# Target counts per stratum
STRATUM_TARGETS = {"A": 150, "B": 100, "C": 100, "D": 50}

# Human-readable stratum descriptions (used in output CSVs and paper)
STRATUM_DESCRIPTIONS = {
    "A": "Temporally extended consequentialist (requires tracking outcomes across steps)",
    "B": "Deontological / rule-based (single applicable rule, no temporal chain)",
    "C": "Virtue / commonsense (character evaluation or social norm)",
    "D": "Mixed / ambiguous (correct framework genuinely contestable)",
}


# ════════════════════════════════════════════════════════════════════════════
# KEYWORD FILTERS
# Applied on top of dataset-category mapping to sharpen stratum membership.
# Stratum A specifically requires scenarios with temporal/causal language.
# ════════════════════════════════════════════════════════════════════════════

# Words that signal temporal extension / consequence chains (Stratum A)
STRATUM_A_KEYWORDS = [
    r"\bif\b.*\bthen\b",          # conditional consequence
    r"\bconsequen",               # consequence, consequences
    r"\bwould\b.*\bhappen\b",     # outcome reasoning
    r"\beventually\b",
    r"\bdownstream\b",
    r"\blong.?term\b",
    r"\bin order to\b",
    r"\bprevent\b",
    r"\bresult\b",
    r"\blead to\b",
    r"\bcause\b",
    r"\bultimately\b",
]

# Words that signal deontological / rule-based reasoning (Stratum B)
STRATUM_B_KEYWORDS = [
    r"\bpromise\b",
    r"\bobligation\b",
    r"\bduty\b",
    r"\bright\b.*\bwrong\b",
    r"\bshould\b",
    r"\bmust\b",
    r"\bprinciple\b",
    r"\brule\b",
    r"\bnever\b",
    r"\balways\b",
]


def keyword_score(text: str, keywords: list) -> int:
    """Count how many keyword patterns match in a scenario text."""
    text = text.lower()
    return sum(1 for kw in keywords if re.search(kw, text))


# ════════════════════════════════════════════════════════════════════════════
# STRATUM BUILDER
# ════════════════════════════════════════════════════════════════════════════

def assign_stratum(row: pd.Series) -> str:
    """
    Assign a stratum label to a dataset item.

    Primary assignment is from STRATUM_MAP (dataset × category).
    For Stratum A items, we additionally verify temporal language is present.
    Items that don't map cleanly go to Stratum D as fallback.
    """
    key = (row["dataset"], row["category"])
    base_stratum = STRATUM_MAP.get(key, "D")

    # Tighten Stratum A: require at least one temporal/causal keyword
    if base_stratum == "A":
        score = keyword_score(str(row["prompt_text"]), STRATUM_A_KEYWORDS)
        if score == 0:
            # Demote to Stratum C if no temporal language found
            return "C"

    return base_stratum


def build_stratified_sample(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Assign strata and sample up to the target count from each stratum.

    Within each stratum, items are sorted by keyword relevance score
    (most relevant first) before sampling, so the top-N most prototypical
    examples are selected.
    """
    df = all_data.copy()
    df["stratum"] = df.apply(assign_stratum, axis=1)
    df["stratum_description"] = df["stratum"].map(STRATUM_DESCRIPTIONS)

    # Compute keyword relevance scores for prioritized sampling
    df["kw_score_A"] = df["prompt_text"].apply(
        lambda t: keyword_score(str(t), STRATUM_A_KEYWORDS)
    )
    df["kw_score_B"] = df["prompt_text"].apply(
        lambda t: keyword_score(str(t), STRATUM_B_KEYWORDS)
    )

    sampled_frames = []

    for stratum, target in STRATUM_TARGETS.items():
        pool = df[df["stratum"] == stratum].copy()

        if len(pool) == 0:
            print(f"  ⚠️  Stratum {stratum}: no items found — check STRATUM_MAP")
            continue

        # Sort by relevance score descending, then sample
        sort_col = "kw_score_A" if stratum == "A" else "kw_score_B"
        pool = pool.sort_values(sort_col, ascending=False)

        n = min(target, len(pool))
        sample = pool.head(n)     # top-N most prototypical items
        sampled_frames.append(sample)

        print(f"  Stratum {stratum}: selected {n}/{target} items "
              f"(pool had {len(pool)})")

    result = pd.concat(sampled_frames, ignore_index=True)

    # Clean up helper columns not needed downstream
    result = result.drop(columns=["kw_score_A", "kw_score_B"], errors="ignore")

    # Assign a clean phase2 item ID
    result["phase2_id"] = [f"p2_{i:04d}" for i in range(len(result))]

    return result


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Load the combined dataset built in Phase 1 Step 1
    raw_path = "data/raw/all_datasets_combined.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"{raw_path} not found.\n"
            "Run Phase 1 → 01_load_datasets.py first."
        )

    all_data = pd.read_csv(raw_path)
    print(f"Loaded {len(all_data)} items from Phase 1 datasets.\n")
    print("Assigning strata and building 400-scenario sample ...")

    sample = build_stratified_sample(all_data)

    # Save the stratified sample
    sample.to_csv("data/phase2/stratified_400.csv", index=False)
    print(f"\n  💾 Saved stratified sample → data/phase2/stratified_400.csv")

    # Save a summary
    summary = (
        sample.groupby(["stratum", "stratum_description"])
        .agg(
            n_items=("phase2_id", "count"),
            datasets=("dataset", lambda x: ", ".join(x.unique())),
        )
        .reset_index()
    )
    summary.to_csv("data/phase2/stratum_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("STRATUM SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nTotal: {len(sample)} scenarios across {sample['stratum'].nunique()} strata")
    print("\nNext step → 02_run_cot_inference.py")
