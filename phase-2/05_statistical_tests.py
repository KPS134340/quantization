"""
PHASE 2 — SCRIPT 5
Statistical Tests.

Runs all four statistical tests specified in the methodology:

  PRIMARY     — Logistic regression: stratum → failure probability
                Confirms the hypothesis: Stratum A has elevated odds ratio
                relative to B and C.

  SECONDARY   — Within Stratum A failures: execution failures (Type 1+3)
                vs. framework switch failures (Type 2).
                If majority are execution failures → mechanism is procedural.

  TERTIARY    — Chain length fingerprint.
                X: FP16 reasoning chain length (n coherent steps, 1–5)
                Y: probability of Type 1 or Type 3 failure under INT4.
                If monotonically increasing → the central figure of the paper.

  QUATERNARY  — Degradation gradient.
                Compare INT8 vs INT4 failure rate curves across chain lengths.
                Shows whether damage is compression-level-dependent.

Outputs
-------
  data/phase2/stats_primary.csv         — logistic regression results
  data/phase2/stats_secondary.csv       — execution vs. framework failure counts
  data/phase2/stats_tertiary.csv        — chain length × failure probability table
  data/phase2/stats_quaternary.csv      — INT8 vs INT4 gradient comparison
  figures/phase2_chain_length_curve.png — the central empirical figure
  figures/phase2_degradation_gradient.png
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

os.makedirs("data/phase2", exist_ok=True)
os.makedirs("figures", exist_ok=True)

MODEL_KEYS = ["llama3", "mistral"]


# ════════════════════════════════════════════════════════════════════════════
# PRIMARY TEST — Logistic Regression: stratum predicts failure
# ════════════════════════════════════════════════════════════════════════════

def primary_test(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Logistic regression: does stratum significantly predict failure probability?

    Predictors: one-hot encoded stratum (A, B, C, D)
    Outcome:    binary failure (1 = any of Types 1–4, 0 = Type 0)
    Reference:  Stratum B (rule-based; lowest expected failure rate)

    Reports odds ratios for each stratum relative to Stratum B.
    Hypothesis predicts OR(Stratum A) >> 1 and significantly > OR(B) = 1.0.
    """
    # Focus on FP16 vs INT4 comparison
    df = pairs_df[pairs_df["comparison"] == "FP16_vs_INT4"].copy()
    df["failure_binary"] = (df["failure_type"] > 0).astype(int)

    # One-hot encode strata with B as reference (dropped)
    dummies = pd.get_dummies(df["stratum"], prefix="stratum", drop_first=False)
    # Drop Stratum B to make it the reference
    if "stratum_B" in dummies.columns:
        dummies = dummies.drop(columns=["stratum_B"])

    X = dummies.values.astype(float)
    y = df["failure_binary"].values

    if X.shape[0] < 20:
        return pd.DataFrame([{"note": "Too few samples for logistic regression"}])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    # Odds ratios = exp(coefficients)
    results = []
    for col, coef in zip(dummies.columns, model.coef_[0]):
        stratum_label = col.replace("stratum_", "")
        n_failures = df[df["stratum"] == stratum_label]["failure_binary"].sum()
        n_total    = (df["stratum"] == stratum_label).sum()

        results.append({
            "stratum":          stratum_label,
            "n_items":          n_total,
            "n_failures":       n_failures,
            "failure_rate":     round(n_failures / n_total, 4) if n_total > 0 else 0,
            "log_odds":         round(coef, 4),
            "odds_ratio":       round(np.exp(coef), 4),
            "interpretation":   (
                f"Stratum {stratum_label} has {np.exp(coef):.2f}x the odds "
                f"of failure compared to Stratum B (reference)"
            ),
        })

    # Add reference row
    ref_n    = (df["stratum"] == "B").sum()
    ref_fail = df[df["stratum"] == "B"]["failure_binary"].sum()
    results.insert(0, {
        "stratum":        "B (reference)",
        "n_items":        ref_n,
        "n_failures":     ref_fail,
        "failure_rate":   round(ref_fail / ref_n, 4) if ref_n > 0 else 0,
        "log_odds":       0.0,
        "odds_ratio":     1.0,
        "interpretation": "Reference category",
    })

    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════════════════
# SECONDARY TEST — Execution vs. Framework failures within Stratum A
# ════════════════════════════════════════════════════════════════════════════

def secondary_test(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Within Stratum A INT4 failures, what proportion are execution failures
    (Type 1 + Type 3) vs framework switch failures (Type 2)?

    If execution failures dominate → model knows what to do but can't execute it
    → directly motivates CRWP protecting procedural pathways.
    """
    stratum_a = pairs_df[
        (pairs_df["stratum"] == "A") &
        (pairs_df["comparison"] == "FP16_vs_INT4") &
        (pairs_df["failure_type"] > 0)
    ].copy()

    if stratum_a.empty:
        return pd.DataFrame([{"note": "No Stratum A failures found"}])

    type_counts = stratum_a["failure_type"].value_counts().reset_index()
    type_counts.columns = ["failure_type", "count"]
    type_counts["failure_label"] = type_counts["failure_type"].map({
        1: "Conclusion drift (execution)",
        2: "Framework switch (recognition)",
        3: "Chain collapse (execution)",
        4: "Full collapse",
    })
    type_counts["proportion"] = (
        type_counts["count"] / type_counts["count"].sum()
    ).round(4)
    type_counts["category"] = type_counts["failure_type"].map(
        lambda t: "Execution failure (Type 1+3)" if t in (1, 3)
        else "Framework failure (Type 2)" if t == 2
        else "Full collapse (Type 4)"
    )

    exec_prop = type_counts[
        type_counts["failure_type"].isin([1, 3])
    ]["proportion"].sum()

    type_counts["note"] = ""
    type_counts.loc[type_counts["failure_type"].isin([1, 3]),
                    "note"] = f"Execution failures: {exec_prop:.1%} of all failures"

    return type_counts


# ════════════════════════════════════════════════════════════════════════════
# TERTIARY TEST — Chain Length Fingerprint (the central figure)
# ════════════════════════════════════════════════════════════════════════════

def tertiary_test(
    pairs_df: pd.DataFrame,
    model_key: str,
) -> pd.DataFrame:
    """
    Plot failure probability as a function of FP16 chain length.

    FP16 chain length = n_coherent_transitions_ref (0–4, mapped to 1–5 steps).
    Y = P(Type 1 or Type 3 failure under INT4) at that chain length.

    If the curve is monotonically increasing → chain length causes failure.
    This is the central empirical finding of Stage 1.
    """
    stratum_a = pairs_df[
        (pairs_df["stratum"] == "A") &
        (pairs_df["comparison"] == "FP16_vs_INT4")
    ].copy()

    if stratum_a.empty:
        return pd.DataFrame()

    # Chain length = number of coherent transitions in FP16 response (0–4)
    # We treat this as a proxy for "reasoning steps required"
    stratum_a["fp16_chain_length"] = stratum_a["n_coherent_transitions_ref"]

    # Execution failure = Type 1 or Type 3
    stratum_a["exec_failure"] = stratum_a["failure_type"].isin([1, 3]).astype(int)

    # Compute failure probability per chain length
    curve = (
        stratum_a.groupby("fp16_chain_length")
        .agg(
            n_items=("exec_failure", "count"),
            n_failures=("exec_failure", "sum"),
            failure_prob=("exec_failure", "mean"),
        )
        .reset_index()
    )
    curve["failure_prob"] = curve["failure_prob"].round(4)

    # Check monotonicity
    probs = curve["failure_prob"].values
    is_monotonic = all(probs[i] <= probs[i+1] for i in range(len(probs)-1))
    curve["monotonic_up_to_here"] = [
        all(probs[:i+1] == sorted(probs[:i+1])) for i in range(len(probs))
    ]

    print(f"\n  Chain length fingerprint ({model_key}):")
    print(curve[["fp16_chain_length", "n_items", "n_failures",
                 "failure_prob"]].to_string(index=False))
    print(f"  Monotonically increasing: {'✅ YES' if is_monotonic else '⚠️ NO'}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        curve["fp16_chain_length"],
        curve["failure_prob"],
        marker="o", color="crimson", linewidth=2.5,
        markersize=8, zorder=5, label="INT4 failure probability"
    )
    ax.fill_between(
        curve["fp16_chain_length"],
        curve["failure_prob"],
        alpha=0.15, color="crimson"
    )

    # Annotate each point with n_items
    for _, row in curve.iterrows():
        ax.annotate(
            f"n={int(row['n_items'])}",
            xy=(row["fp16_chain_length"], row["failure_prob"]),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=8, color="crimson",
        )

    ax.set_xlabel("FP16 Chain Length\n(# coherent reasoning steps in reference model)",
                  fontsize=10)
    ax.set_ylabel("P(Type 1 or Type 3 failure under INT4)", fontsize=10)
    ax.set_title(
        f"Chain Length Fingerprint — {model_key.upper()}\n"
        f"Stratum A scenarios only. "
        f"{'Monotonically increasing ✓' if is_monotonic else 'Non-monotonic'}",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylim(0, 1.05)
    ax.set_xticks(curve["fp16_chain_length"])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = f"figures/phase2_chain_length_curve_{model_key}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved chain length figure → {out_path}")

    return curve


# ════════════════════════════════════════════════════════════════════════════
# QUATERNARY TEST — Degradation Gradient (INT8 vs INT4)
# ════════════════════════════════════════════════════════════════════════════

def quaternary_test(pairs_df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """
    Compare INT8 vs INT4 failure rate curves across chain lengths.

    If INT4 shows a steeper curve than INT8, compression level specifically
    damages long-chain reasoning — directly motivating CRWP's mixed-precision
    approach (protecting procedural weights in FP16/INT8, compressing others).
    """
    stratum_a = pairs_df[
        (pairs_df["stratum"] == "A") &
        (pairs_df["comparison"].isin(["FP16_vs_INT8", "FP16_vs_INT4"]))
    ].copy()

    if stratum_a.empty:
        return pd.DataFrame()

    stratum_a["fp16_chain_length"] = stratum_a["n_coherent_transitions_ref"]
    stratum_a["exec_failure"] = stratum_a["failure_type"].isin([1, 3]).astype(int)

    gradient = (
        stratum_a.groupby(["fp16_chain_length", "comparison"])
        .agg(failure_prob=("exec_failure", "mean"), n_items=("exec_failure", "count"))
        .reset_index()
    )
    gradient["failure_prob"] = gradient["failure_prob"].round(4)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"FP16_vs_INT8": "steelblue", "FP16_vs_INT4": "crimson"}
    labels = {"FP16_vs_INT8": "INT8 (mild compression)", "FP16_vs_INT4": "INT4 (aggressive)"}

    for cmp, group in gradient.groupby("comparison"):
        ax.plot(
            group["fp16_chain_length"],
            group["failure_prob"],
            marker="o", color=colors.get(cmp, "grey"),
            linewidth=2.5, markersize=8,
            label=labels.get(cmp, cmp)
        )

    ax.set_xlabel("FP16 Chain Length (# coherent steps)", fontsize=10)
    ax.set_ylabel("P(Execution failure)", fontsize=10)
    ax.set_title(
        f"Degradation Gradient — {model_key.upper()}\n"
        "INT8 vs INT4 failure rates across chain lengths",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = f"figures/phase2_degradation_gradient_{model_key}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved degradation gradient → {out_path}")

    return gradient


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for model_key in MODEL_KEYS:
        pairs_path = f"data/phase2/failure_pairs_{model_key}.csv"
        if not os.path.exists(pairs_path):
            print(f"⚠️  {pairs_path} not found — run 04_failure_taxonomy.py first.")
            continue

        pairs = pd.read_csv(pairs_path)
        print(f"\n{'='*60}")
        print(f"STATISTICAL TESTS — {model_key.upper()}")
        print(f"{'='*60}")

        # Primary
        primary = primary_test(pairs)
        primary.to_csv(f"data/phase2/stats_primary_{model_key}.csv", index=False)
        print("\nPRIMARY — Logistic Regression Odds Ratios:")
        print(primary[["stratum", "n_items", "failure_rate",
                        "odds_ratio"]].to_string(index=False))

        # Secondary
        secondary = secondary_test(pairs)
        secondary.to_csv(f"data/phase2/stats_secondary_{model_key}.csv", index=False)
        print("\nSECONDARY — Execution vs Framework Failures (Stratum A):")
        print(secondary[["failure_label", "count", "proportion"]].to_string(index=False))

        # Tertiary
        tertiary = tertiary_test(pairs, model_key)
        tertiary.to_csv(f"data/phase2/stats_tertiary_{model_key}.csv", index=False)

        # Quaternary
        quaternary = quaternary_test(pairs, model_key)
        quaternary.to_csv(f"data/phase2/stats_quaternary_{model_key}.csv", index=False)

    print("\n✅ All statistical tests complete. Next step → 06_qualitative_cases.py")
