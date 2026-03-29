"""
PHASE 2 — SCRIPT 4
Failure Taxonomy Classification.

For every FP16 vs INT4 (and FP16 vs INT8) pair, this script classifies
the comparison into one of five failure types from the methodology:

  Type 0 — No failure       Same conclusion, same framework, coherence comparable
  Type 1 — Conclusion drift Different conclusion, same framework applied
  Type 2 — Framework switch Different framework applied (FP16 vs INT4)
  Type 3 — Chain collapse   Framework may match; coherence drops sharply mid-chain
  Type 4 — Full collapse    Incoherent output, circular reasoning, refusal

The evidential hierarchy (from methodology):
  Type 3 in Stratum A → strongest evidence for QCMST hypothesis
  Type 1 in Stratum A → strong evidence
  Any failure in A >> B/C → sufficient for clustering claim
  Type 2 in Stratum A → weakest (supplementary)

Outputs
-------
  data/phase2/failure_pairs_<model_key>.csv   — one row per (item × comparison)
  data/phase2/failure_summary_<model_key>.csv — counts per (stratum × failure type)
  data/phase2/token_flip_log_<model_key>.csv  — first failure step per Stratum A item
                                                 (consumed by Stage 3 / CRWP)
"""

import pandas as pd
import numpy as np
import os

os.makedirs("data/phase2", exist_ok=True)

MODEL_KEYS = ["llama3", "mistral"]

# Threshold: coherence drop of this magnitude = "sharp drop" → Type 3
COHERENCE_DROP_THRESHOLD = 0.25

# Coherence score below this in INT4 = "chain collapse"
COHERENCE_COLLAPSE_THRESHOLD = 0.50


# ════════════════════════════════════════════════════════════════════════════
# PAIR BUILDER
# Join FP16 and compressed-precision rows by phase2_id to create pairs
# ════════════════════════════════════════════════════════════════════════════

def build_pairs(
    annotations_df: pd.DataFrame,
    reference: str = "FP16",
    comparisons: list = ("INT8", "INT4"),
) -> pd.DataFrame:
    """
    Create a paired DataFrame: one row per (scenario × comparison precision).

    Each row has _ref columns (FP16 values) and _cmp columns (INT8/INT4 values).
    """
    ref = annotations_df[annotations_df["precision"] == reference].copy()
    ref = ref.add_suffix("_ref").rename(
        columns={"phase2_id_ref": "phase2_id", "stratum_ref": "stratum"}
    )

    all_pairs = []

    for cmp_precision in comparisons:
        cmp = annotations_df[annotations_df["precision"] == cmp_precision].copy()
        cmp = cmp.add_suffix("_cmp")

        merged = ref.merge(
            cmp,
            left_on="phase2_id",
            right_on="phase2_id_cmp",
            how="inner",
        )
        merged["comparison"] = f"{reference}_vs_{cmp_precision}"
        all_pairs.append(merged)

    return pd.concat(all_pairs, ignore_index=True)


# ════════════════════════════════════════════════════════════════════════════
# FAILURE TYPE CLASSIFIER
# Applied row-by-row to the paired DataFrame
# ════════════════════════════════════════════════════════════════════════════

def classify_failure_type(row: pd.Series) -> int:
    """
    Classify one FP16 vs compressed-model pair into failure type 0–4.

    Decision logic (applied in order — first matching type wins):

    Type 4 — Full collapse
      INT4's final_judgment is 'unclear' (model gave incoherent / no answer)
      OR coherence score is 0.0 (every transition failed)

    Type 3 — Chain collapse
      Framework matches BUT coherence drops sharply OR collapses below threshold
      The reasoning broke down mid-chain even though the framework was recognized

    Type 1 — Conclusion drift
      Framework matches, coherence is acceptable, BUT final judgment differs
      Model knows what to do but reached the wrong conclusion

    Type 2 — Framework switch
      Framework changed between FP16 and INT4
      Methodological note: this is the weakest signal — see methodology doc

    Type 0 — No failure
      Same framework, same (or similar) conclusion, comparable coherence
    """
    fw_ref  = str(row.get("framework_ref",  "Unclear")).strip()
    fw_cmp  = str(row.get("framework_cmp",  "Unclear")).strip()
    jdg_ref = str(row.get("final_judgment_ref", "unclear")).strip()
    jdg_cmp = str(row.get("final_judgment_cmp", "unclear")).strip()
    coh_ref = float(row.get("chain_coherence_score_ref", 1.0))
    coh_cmp = float(row.get("chain_coherence_score_cmp", 1.0))

    coh_drop = coh_ref - coh_cmp

    # ── Type 4: Full collapse ─────────────────────────────────────────────
    if jdg_cmp == "unclear" or coh_cmp == 0.0:
        return 4

    # ── Type 3: Chain collapse ────────────────────────────────────────────
    # Framework may match OR may differ; the defining feature is
    # a sharp coherence drop in the compressed model
    if (coh_drop >= COHERENCE_DROP_THRESHOLD or
            coh_cmp < COHERENCE_COLLAPSE_THRESHOLD):
        return 3

    # ── Type 1: Conclusion drift (framework matches, conclusion doesn't) ──
    framework_matches = (fw_ref == fw_cmp) or (
        fw_ref in ("Unclear", "Mixed") or fw_cmp in ("Unclear", "Mixed")
    )
    conclusion_differs = (
        jdg_ref != jdg_cmp and
        jdg_ref != "unclear" and
        jdg_cmp != "unclear"
    )

    if framework_matches and conclusion_differs:
        return 1

    # ── Type 2: Framework switch ──────────────────────────────────────────
    if not framework_matches:
        return 2

    # ── Type 0: No failure ────────────────────────────────────────────────
    return 0


def get_failure_label(failure_type: int) -> str:
    return {
        0: "No failure",
        1: "Conclusion drift",
        2: "Framework switch",
        3: "Chain collapse",
        4: "Full collapse",
    }.get(failure_type, "Unknown")


def is_execution_failure(failure_type: int) -> bool:
    """Types 1 and 3 are execution failures (vs Type 2 recognition failure)."""
    return failure_type in (1, 3)


# ════════════════════════════════════════════════════════════════════════════
# TOKEN FLIP EVENT LOG
# For Stratum A failures: record which transition first failed.
# This is the input to Stage 3 (CRWP) — it identifies which attention
# heads / MLP layers mediate the broken transitions.
# ════════════════════════════════════════════════════════════════════════════

def build_token_flip_log(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the token-flip event log for Stratum A failures.

    For each Stratum A item that has a Type 1, 3, or 4 failure under INT4,
    record:
      - phase2_id
      - first_failure_transition (from INT4 annotation)
      - failure_type
      - chain_coherence_score (FP16 vs INT4)

    This table is the direct input to Stage 3 CRWP weight identification.
    """
    stratum_a_int4 = pairs_df[
        (pairs_df["stratum"] == "A") &
        (pairs_df["comparison"] == "FP16_vs_INT4") &
        (pairs_df["failure_type"].isin([1, 3, 4]))
    ].copy()

    log = stratum_a_int4[[
        "phase2_id",
        "failure_type",
        "failure_label",
        "first_failure_transition_cmp",
        "chain_coherence_score_ref",
        "chain_coherence_score_cmp",
        "framework_ref",
        "framework_cmp",
    ]].rename(columns={
        "first_failure_transition_cmp": "first_failure_transition",
        "chain_coherence_score_ref":    "fp16_coherence",
        "chain_coherence_score_cmp":    "int4_coherence",
        "framework_ref":                "fp16_framework",
        "framework_cmp":                "int4_framework",
    })

    return log


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# Failure type counts per (stratum × comparison × failure type)
# ════════════════════════════════════════════════════════════════════════════

def build_failure_summary(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the taxonomy table from the methodology:
    rows = stratum × comparison, columns = failure type counts + rates.
    """
    summary = (
        pairs_df.groupby(["stratum", "comparison", "failure_type", "failure_label"])
        .size()
        .reset_index(name="count")
    )

    totals = (
        pairs_df.groupby(["stratum", "comparison"])
        .size()
        .reset_index(name="total")
    )

    summary = summary.merge(totals, on=["stratum", "comparison"])
    summary["rate"] = (summary["count"] / summary["total"]).round(4)

    # Overall failure rate per stratum (Types 1+2+3+4)
    failure_rate = (
        pairs_df[pairs_df["failure_type"] > 0]
        .groupby(["stratum", "comparison"])
        .size()
        .reset_index(name="n_failures")
    )
    failure_rate = failure_rate.merge(totals, on=["stratum", "comparison"])
    failure_rate["overall_failure_rate"] = (
        failure_rate["n_failures"] / failure_rate["total"]
    ).round(4)

    return summary, failure_rate


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for model_key in MODEL_KEYS:
        ann_path = f"data/phase2/annotations_{model_key}.csv"
        if not os.path.exists(ann_path):
            print(f"⚠️  {ann_path} not found — run 03_classify_frameworks.py first.")
            continue

        annotations = pd.read_csv(ann_path)
        print(f"\n{'='*60}")
        print(f"FAILURE TAXONOMY — {model_key.upper()}")
        print(f"{'='*60}")

        # Build FP16 vs INT8 and FP16 vs INT4 pairs
        pairs = build_pairs(annotations)

        # Classify each pair
        pairs["failure_type"]       = pairs.apply(classify_failure_type, axis=1)
        pairs["failure_label"]      = pairs["failure_type"].apply(get_failure_label)
        pairs["is_execution_failure"] = pairs["failure_type"].apply(is_execution_failure)

        # Save full pairs table
        pairs_path = f"data/phase2/failure_pairs_{model_key}.csv"
        pairs.to_csv(pairs_path, index=False)
        print(f"\n  💾 Saved failure pairs → {pairs_path}")

        # Build and save summary
        summary, failure_rate = build_failure_summary(pairs)
        summary_path = f"data/phase2/failure_summary_{model_key}.csv"
        summary.to_csv(summary_path, index=False)

        print("\nFailure rate by stratum (FP16 vs INT4):")
        int4_rate = failure_rate[failure_rate["comparison"] == "FP16_vs_INT4"]
        print(int4_rate[["stratum", "n_failures", "total",
                          "overall_failure_rate"]].to_string(index=False))

        # Build and save token-flip event log
        flip_log = build_token_flip_log(pairs)
        flip_path = f"data/phase2/token_flip_log_{model_key}.csv"
        flip_log.to_csv(flip_path, index=False)
        print(f"\n  💾 Saved token-flip log ({len(flip_log)} Stratum A failures)"
              f" → {flip_path}")
        print(f"     (This file feeds into Stage 3 / CRWP)")

    print("\nTaxonomy complete. Next step → 05_statistical_tests.py")
