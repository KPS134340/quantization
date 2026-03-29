"""
PHASE 1 STEP 1 — Load all four datasets and preview their structure.

Datasets used:
  - ETHICS       (Hendrycks et al., 2021) — multiple splits
  - MoralBench   (Ji et al., 2024)
  - Moral Stories (Emelin et al., 2021)
  - ValueBench   (Value Kaleidoscope)
"""

from datasets import load_dataset
import pandas as pd
import os

# ── Output directory ──────────────────────────────────────────────────────────
os.makedirs("data/raw", exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. ETHICS DATASET
#    Five splits: commonsense, deontology, justice, utilitarianism, virtue
# ════════════════════════════════════════════════════════════════════════════

def load_ethics():
    splits = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]
    frames = []

    for split_name in splits:
        print(f"Loading ETHICS / {split_name} ...")
        ds = load_dataset("hendrycks/ethics", split_name, split="test")

        # Each ETHICS item has an 'input' (scenario text) and 'label' (0 or 1)
        # Label meanings: 1 = morally acceptable / right, 0 = not acceptable
        for item in ds:
            frames.append({
                "dataset":   "ETHICS",
                "category":  split_name,
                "item_id":   f"ethics_{split_name}_{item.get('index', len(frames))}",
                # Unify the text field — ETHICS uses 'input' or 'scenario'
                "prompt_text": item.get("input") or item.get("scenario", ""),
                "gold_label": str(item.get("label", "")),
                # We'll build the actual model prompt in 03_run_inference.py
                "answer_options": "0,1",   # 0 = wrong/unacceptable, 1 = right/acceptable
            })

    df = pd.DataFrame(frames)
    df.to_csv("data/raw/ethics.csv", index=False)
    print(f"  Saved {len(df)} ETHICS items → data/raw/ethics.csv\n")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. MORALBENCH
#    Scenario-level moral judgment; richer context than ETHICS
# ════════════════════════════════════════════════════════════════════════════

def load_moralbench():
    print("Loading MoralBench ...")
    # MoralBench is hosted under "MoralBench/MoralBench" on HuggingFace
    ds = load_dataset("MoralBench/MoralBench", split="test")

    frames = []
    for i, item in enumerate(ds):
        # MoralBench items typically have 'scenario' and multiple choice options
        frames.append({
            "dataset":      "MoralBench",
            "category":     item.get("category", "general"),
            "item_id":      f"moralbench_{i}",
            "prompt_text":  item.get("scenario", item.get("question", "")),
            "gold_label":   str(item.get("answer", item.get("label", ""))),
            "answer_options": item.get("choices", "A,B,C,D"),
        })

    df = pd.DataFrame(frames)
    df.to_csv("data/raw/moralbench.csv", index=False)
    print(f"  Saved {len(df)} MoralBench items → data/raw/moralbench.csv\n")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 3. MORAL STORIES
#    Situation → intention → action → consequence structure
#    We use the binary acceptability judgment format for Phase 1
# ════════════════════════════════════════════════════════════════════════════

def load_moral_stories():
    print("Loading Moral Stories ...")
    ds = load_dataset("demelin/moral_stories", "full", split="test")

    frames = []
    for i, item in enumerate(ds):
        # Moral Stories has: situation, intention, moral_action, immoral_action,
        # moral_consequence, immoral_consequence
        # For Phase 1 we present the situation + one action and ask: acceptable? yes/no
        # We create two rows per story: one moral, one immoral (counterbalanced)
        for action_type in ["moral", "immoral"]:
            action_text = item.get(f"{action_type}_action", "")
            gold        = "yes" if action_type == "moral" else "no"
            frames.append({
                "dataset":      "MoralStories",
                "category":     "acceptability",
                "item_id":      f"moralstories_{i}_{action_type}",
                "prompt_text":  (
                    f"Situation: {item.get('situation', '')}\n"
                    f"Intention: {item.get('intention', '')}\n"
                    f"Action: {action_text}"
                ),
                "gold_label":   gold,
                "answer_options": "yes,no",
            })

    df = pd.DataFrame(frames)
    # Cap at 2000 items to keep inference runtime manageable
    df = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)
    df.to_csv("data/raw/moral_stories.csv", index=False)
    print(f"  Saved {len(df)} Moral Stories items → data/raw/moral_stories.csv\n")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 4. VALUEBENCH (Value Kaleidoscope)
#    Schwartz Value Theory scaffolded items — framework classification
# ════════════════════════════════════════════════════════════════════════════

def load_valuebench():
    print("Loading ValueBench ...")
    # Dataset identifier may vary; common HuggingFace path:
    ds = load_dataset("value-kaleidoscope/ValueBench", split="test")

    frames = []
    for i, item in enumerate(ds):
        frames.append({
            "dataset":      "ValueBench",
            "category":     item.get("value_category", "unknown"),
            "item_id":      f"valuebench_{i}",
            "prompt_text":  item.get("scenario", item.get("text", "")),
            "gold_label":   str(item.get("label", item.get("answer", ""))),
            "answer_options": item.get("options", "A,B"),
        })

    df = pd.DataFrame(frames)
    df.to_csv("data/raw/valuebench.csv", index=False)
    print(f"  Saved {len(df)} ValueBench items → data/raw/valuebench.csv\n")
    return df


# ════════════════════════════════════════════════════════════════════════════
# MAIN — load all four and print a summary
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ethics        = load_ethics()
    moralbench    = load_moralbench()
    moral_stories = load_moral_stories()
    valuebench    = load_valuebench()

    all_data = pd.concat([ethics, moralbench, moral_stories, valuebench],
                         ignore_index=True)
    all_data.to_csv("data/raw/all_datasets_combined.csv", index=False)

    print("=" * 55)
    print("DATASET SUMMARY")
    print("=" * 55)
    summary = all_data.groupby("dataset")["item_id"].count().reset_index()
    summary.columns = ["Dataset", "Item Count"]
    print(summary.to_string(index=False))
    print(f"\nTotal items: {len(all_data)}")
    print("\nNext step → run 02_load_models.py")
