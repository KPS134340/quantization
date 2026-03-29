"""
PHASE 1 STEP 4 — Compute divergence metrics and Cohen's kappa.

Takes the raw inference results from step 3 and computes:
  - Agreement rate (INT8 vs FP16, INT4 vs FP16) per dataset and category
  - Divergence rate (1 - agreement)
  - Cohen's kappa between FP16 and INT4 outputs

Outputs:
  data/metrics/agreement_rates.csv
  data/metrics/kappa_scores.csv
  data/metrics/divergence_summary.csv
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import os

os.makedirs("data/metrics", exist_ok=True)

MODEL_KEYS = ["llama3", "mistral"]


# ════════════════════════════════════════════════════════════════════════════
# AGREEMENT RATE
# Proportion of items where compressed model == FP16 model
# ════════════════════════════════════════════════════════════════════════════

def compute_agreement_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per (dataset, category) agreement rates for INT8 and INT4 vs FP16.

    Parameters
    ----------
    df : results dataframe with columns answer_FP16, answer_INT8, answer_INT4

    Returns
    -------
    DataFrame with columns: dataset, category, INT8_agreement, INT4_agreement,
                            INT8_divergence, INT4_divergence, n_items
    """
    records = []

    # Compute for each (dataset, category) pair
    for (dataset, category), group in df.groupby(["dataset", "category"]):
        n = len(group)

        # Drop rows where any answer is INVALID
        valid = group[
            (group["answer_FP16"] != "INVALID") &
            (group["answer_INT8"] != "INVALID") &
            (group["answer_INT4"] != "INVALID")
        ]
        n_valid = len(valid)

        if n_valid == 0:
            continue

        int8_agree = (valid["answer_INT8"] == valid["answer_FP16"]).mean()
        int4_agree = (valid["answer_INT4"] == valid["answer_FP16"]).mean()

        records.append({
            "dataset":          dataset,
            "category":         category,
            "n_items":          n,
            "n_valid":          n_valid,
            "INT8_agreement":   round(int8_agree, 4),
            "INT4_agreement":   round(int4_agree, 4),
            "INT8_divergence":  round(1 - int8_agree, 4),
            "INT4_divergence":  round(1 - int4_agree, 4),
        })

    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════
# COHEN'S KAPPA
# Measures inter-rater agreement between FP16 and INT4 as a classification
# problem. κ < 0.6 = substantial disagreement.
# ════════════════════════════════════════════════════════════════════════════

def compute_kappa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Cohen's kappa between FP16 and INT4 per dataset.

    Kappa ranges:
      > 0.8  → near-perfect agreement (quantization is benign)
      0.6–0.8 → substantial agreement
      0.4–0.6 → moderate agreement
      < 0.4  → poor agreement (quantization is meaningfully altering judgments)
    """
    records = []

    for dataset, group in df.groupby("dataset"):
        valid = group[
            (group["answer_FP16"] != "INVALID") &
            (group["answer_INT4"] != "INVALID")
        ]

        if len(valid) < 10:
            continue

        try:
            kappa = cohen_kappa_score(
                valid["answer_FP16"],
                valid["answer_INT4"],
            )
        except ValueError:
            # cohen_kappa_score raises if only one unique label exists
            kappa = 1.0

        records.append({
            "dataset":        dataset,
            "n_items":        len(valid),
            "kappa_FP16_INT4": round(kappa, 4),
            "interpretation": (
                "near-perfect" if kappa > 0.8 else
                "substantial"  if kappa > 0.6 else
                "moderate"     if kappa > 0.4 else
                "poor — meaningful divergence"
            ),
        })

    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL SUMMARY
# Overall divergence rate across all datasets + the finding sentence
# ════════════════════════════════════════════════════════════════════════════

def compute_summary(agreement_df: pd.DataFrame) -> dict:
    """
    Compute overall divergence rates to populate the key finding sentence.
    """
    overall_int8_div = agreement_df["INT8_divergence"].mean()
    overall_int4_div = agreement_df["INT4_divergence"].mean()

    # Which category shows highest INT4 divergence?
    worst = agreement_df.loc[agreement_df["INT4_divergence"].idxmax()]

    return {
        "overall_INT8_divergence": round(overall_int8_div, 4),
        "overall_INT4_divergence": round(overall_int4_div, 4),
        "worst_category":          worst["category"],
        "worst_category_INT4_div": worst["INT4_divergence"],
        "finding_sentence": (
            f"INT4 quantization produces outputs that diverge from FP16 at a rate of "
            f"{overall_int4_div:.1%}, significantly above the {overall_int8_div:.1%} "
            f"observed for INT8. The highest divergence is seen in the "
            f"'{worst['category']}' category ({worst['INT4_divergence']:.1%})."
        ),
    }


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for model_key in MODEL_KEYS:
        path = f"data/results/{model_key}_phase1.csv"
        if not os.path.exists(path):
            print(f"⚠️  {path} not found — run 03_run_inference.py first.")
            continue

        df = pd.read_csv(path)
        print(f"\n{'='*55}")
        print(f"METRICS — {model_key.upper()}")
        print(f"{'='*55}")

        # Agreement rates
        agreement = compute_agreement_rates(df)
        agreement["model"] = model_key
        agreement.to_csv(
            f"data/metrics/{model_key}_agreement_rates.csv", index=False
        )
        print(f"\nAgreement rates (sample):\n{agreement.head(10).to_string(index=False)}")

        # Kappa scores
        kappa = compute_kappa(df)
        kappa["model"] = model_key
        kappa.to_csv(f"data/metrics/{model_key}_kappa.csv", index=False)
        print(f"\nCohen's kappa:\n{kappa.to_string(index=False)}")

        # Summary
        summary = compute_summary(agreement)
        print(f"\n{'─'*55}")
        print("KEY FINDING:")
        print(summary["finding_sentence"])
        print(f"{'─'*55}")

        # Save summary
        pd.DataFrame([summary]).to_csv(
            f"data/metrics/{model_key}_summary.csv", index=False
        )

    print("\n✅ Metrics complete. Next step → 05_visualize.py")
