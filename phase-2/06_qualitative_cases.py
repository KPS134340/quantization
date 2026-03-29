"""
PHASE 2 — SCRIPT 6
Qualitative Case Selection.

Selects approximately 30 high-quality Type 1 and Type 3 failure examples
from Stratum A for inclusion in the paper's qualitative evidence section
(Section 3.2 of the paper).

Selection criteria (from methodology):
  - Must be Stratum A (temporally extended consequentialist)
  - Must be Type 1 (conclusion drift) or Type 3 (chain collapse)
  - FP16 chain coherence score ≥ 0.75 (FP16 responded well)
  - INT4 chain coherence score ≤ 0.50 (INT4 broke down)
  - Prefer cases where the failure transition is clearly identifiable
  - Prefer cases where the scenario text is self-contained (readable alone)

Outputs
-------
  data/phase2/qualitative_cases.csv    — 30 selected cases
  data/phase2/qualitative_display.txt  — formatted for easy reading / paper appendix
"""

import pandas as pd
import os

os.makedirs("data/phase2", exist_ok=True)

MODEL_KEYS  = ["llama3", "mistral"]
TARGET_CASES = 30  # methodology specifies ~30


# ════════════════════════════════════════════════════════════════════════════
# CASE SELECTOR
# ════════════════════════════════════════════════════════════════════════════

def select_qualitative_cases(
    pairs_df: pd.DataFrame,
    cot_df: pd.DataFrame,
    target: int = TARGET_CASES,
) -> pd.DataFrame:
    """
    Select the best qualitative cases for the paper.

    Parameters
    ----------
    pairs_df : failure pairs table (from 04_failure_taxonomy.py)
    cot_df   : full CoT outputs table (from 02_run_cot_inference.py)
    target   : number of cases to select

    Returns
    -------
    DataFrame with scenario text, FP16 output, INT4 output, failure type
    """
    # Filter to Stratum A, INT4 comparison, Type 1 and Type 3 only
    candidates = pairs_df[
        (pairs_df["stratum"] == "A") &
        (pairs_df["comparison"] == "FP16_vs_INT4") &
        (pairs_df["failure_type"].isin([1, 3]))
    ].copy()

    if candidates.empty:
        print("  ⚠️  No qualifying cases found.")
        return pd.DataFrame()

    # Quality filters
    candidates = candidates[
        (candidates["chain_coherence_score_ref"] >= 0.75) &
        (candidates["chain_coherence_score_cmp"] <= 0.50)
    ]

    # Prioritize cases where the failure transition is clearly identified
    candidates["has_clear_failure"] = (
        candidates["first_failure_transition_cmp"] != "none"
    )

    # Compute a "case quality score" for ranking
    candidates["quality_score"] = (
        candidates["chain_coherence_score_ref"]          # FP16 was coherent
        - candidates["chain_coherence_score_cmp"]         # INT4 was not
        + candidates["has_clear_failure"].astype(float)   # failure is identifiable
    )

    # Sort by quality score descending; alternate Type 1 and Type 3
    # to get a balanced set
    type1 = candidates[candidates["failure_type"] == 1].nlargest(
        target // 2, "quality_score"
    )
    type3 = candidates[candidates["failure_type"] == 3].nlargest(
        target - len(type1), "quality_score"
    )
    selected = pd.concat([type1, type3]).sort_values(
        "quality_score", ascending=False
    ).head(target)

    # Enrich with the actual text from the CoT output table
    # Merge FP16 outputs
    fp16_cot = cot_df[cot_df["precision"] == "FP16"][[
        "phase2_id", "prompt_text", "full_output",
        "step_1", "step_2", "step_3", "step_4", "step_5",
    ]].add_suffix("_fp16").rename(columns={"phase2_id_fp16": "phase2_id"})

    # Merge INT4 outputs
    int4_cot = cot_df[cot_df["precision"] == "INT4"][[
        "phase2_id", "full_output",
        "step_1", "step_2", "step_3", "step_4", "step_5",
    ]].add_suffix("_int4").rename(columns={"phase2_id_int4": "phase2_id"})

    result = (
        selected
        .merge(fp16_cot, on="phase2_id", how="left")
        .merge(int4_cot, on="phase2_id", how="left")
    )

    cols = [
        "phase2_id", "stratum", "failure_type", "failure_label",
        "quality_score",
        "chain_coherence_score_ref", "chain_coherence_score_cmp",
        "first_failure_transition_cmp",
        "framework_ref", "framework_cmp",
        "prompt_text_fp16",
        "step_1_fp16", "step_2_fp16", "step_3_fp16", "step_4_fp16", "step_5_fp16",
        "step_1_int4", "step_2_int4", "step_3_int4", "step_4_int4", "step_5_int4",
    ]
    return result[[c for c in cols if c in result.columns]]


# ════════════════════════════════════════════════════════════════════════════
# DISPLAY FORMATTER
# Formats selected cases for easy reading and paper appendix inclusion
# ════════════════════════════════════════════════════════════════════════════

def format_case_for_display(row: pd.Series, case_num: int) -> str:
    """
    Render a single case in a structured, readable format.
    """
    divider = "═" * 70
    thin    = "─" * 70

    lines = [
        "",
        divider,
        f"CASE {case_num:02d}  |  ID: {row['phase2_id']}  |  "
        f"Failure: {row['failure_label']} (Type {int(row['failure_type'])})",
        f"FP16 coherence: {row['chain_coherence_score_ref']:.2f}  "
        f"INT4 coherence: {row['chain_coherence_score_cmp']:.2f}  "
        f"First failure: {row.get('first_failure_transition_cmp', 'N/A')}",
        divider,
        "",
        "SCENARIO:",
        str(row.get("prompt_text_fp16", ""))[:600],
        "",
        thin,
        "FP16 RESPONSE (reference — coherent):",
        thin,
        f"Step 1: {str(row.get('step_1_fp16', ''))[:300]}",
        f"Step 2: {str(row.get('step_2_fp16', ''))[:300]}",
        f"Step 3 [{row.get('framework_ref', '?')}]: {str(row.get('step_3_fp16', ''))[:300]}",
        f"Step 4: {str(row.get('step_4_fp16', ''))[:300]}",
        f"Step 5: {str(row.get('step_5_fp16', ''))[:200]}",
        "",
        thin,
        "INT4 RESPONSE (compressed — breakdown):",
        thin,
        f"Step 1: {str(row.get('step_1_int4', ''))[:300]}",
        f"Step 2: {str(row.get('step_2_int4', ''))[:300]}",
        f"Step 3 [{row.get('framework_cmp', '?')}]: {str(row.get('step_3_int4', ''))[:300]}",
        f"Step 4: {str(row.get('step_4_int4', ''))[:300]}",
        f"Step 5: {str(row.get('step_5_int4', ''))[:200]}",
        "",
    ]
    return "\n".join(lines)


def write_display_file(cases_df: pd.DataFrame, out_path: str):
    """Write all formatted cases to a single readable text file."""
    lines = [
        "PHASE 2 — QUALITATIVE CASES",
        "Type 1 (Conclusion Drift) and Type 3 (Chain Collapse) failures",
        "Stratum A (Temporally Extended Consequentialist) scenarios only",
        f"Total cases: {len(cases_df)}",
        "",
    ]
    for i, (_, row) in enumerate(cases_df.iterrows(), start=1):
        lines.append(format_case_for_display(row, i))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  💾 Saved display file → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for model_key in MODEL_KEYS:
        pairs_path = f"data/phase2/failure_pairs_{model_key}.csv"
        cot_path   = f"data/phase2/cot_outputs_{model_key}.csv"

        if not os.path.exists(pairs_path) or not os.path.exists(cot_path):
            print(f"⚠️  Missing files for {model_key}. "
                  "Run scripts 02–04 first.")
            continue

        pairs_df = pd.read_csv(pairs_path)
        cot_df   = pd.read_csv(cot_path)

        # Load the stratified sample to get prompt_text
        sample_df = pd.read_csv("data/phase2/stratified_400.csv")
        cot_df    = cot_df.merge(
            sample_df[["phase2_id", "prompt_text"]],
            on="phase2_id", how="left"
        )

        print(f"\nSelecting qualitative cases for {model_key.upper()} ...")
        cases = select_qualitative_cases(pairs_df, cot_df)

        if cases.empty:
            continue

        # Save CSV
        csv_path = f"data/phase2/qualitative_cases_{model_key}.csv"
        cases.to_csv(csv_path, index=False)
        print(f"  💾 Saved {len(cases)} cases → {csv_path}")

        # Save formatted display file
        txt_path = f"data/phase2/qualitative_display_{model_key}.txt"
        write_display_file(cases, txt_path)

        print(f"\n  Summary for {model_key.upper()}:")
        print(f"  Type 1 cases: {(cases['failure_type'] == 1).sum()}")
        print(f"  Type 3 cases: {(cases['failure_type'] == 3).sum()}")
        print(f"  Mean FP16 coherence: "
              f"{cases['chain_coherence_score_ref'].mean():.3f}")
        print(f"  Mean INT4 coherence: "
              f"{cases['chain_coherence_score_cmp'].mean():.3f}")

    print("\n✅ Qualitative cases complete. Phase 2 pipeline done.")
    print("\nOutputs ready for the paper:")
    print("  data/phase2/qualitative_cases_*.csv       → paper appendix table")
    print("  data/phase2/qualitative_display_*.txt     → paper Section 3.2")
    print("  figures/phase2_chain_length_curve_*.png   → paper Figure 2")
    print("  figures/phase2_degradation_gradient_*.png → paper Figure 3")
    print("  data/phase2/stats_primary_*.csv            → paper Table 1")
    print("  data/phase2/token_flip_log_*.csv           → Stage 3 / CRWP input")
