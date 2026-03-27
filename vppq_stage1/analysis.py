import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def analyze_phase1(filepath: str, output_dir: str = "plots"):
    \"\"\"Generates Divergence Heatmap for Phase 1.\"\"\"
    if not os.path.exists(filepath): return
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(filepath)
    df['int8_div'] = 1.0 - df['fp16_int8_match']
    df['int4_div'] = 1.0 - df['fp16_int4_match']
    
    summary = df.groupby(['dataset', 'category'])[['int8_div', 'int4_div']].mean().reset_index()
    summary['label'] = summary['dataset'] + " - " + summary['category']
    
    pivot = summary.set_index('label')[['int8_div', 'int4_div']]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2%", cmap="Reds", vmin=0, vmax=max(pivot.values.max(), 0.5))
    plt.title("Phase 1: Output Divergence Rate Map by Category")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase1_divergence_heatmap.png"))
    plt.close()
    print("Saved Phase 1 Heatmap.")

def analyze_phase2(filepath: str, output_dir: str = "plots"):
    \"\"\"
    Generates mechanstic taxonomy and chain-length figures for Phase 2.
    \"\"\"
    if not os.path.exists(filepath): return
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(filepath)
    
    # 1. Failure Taxonomy bar plot
    stratum_groups = df.groupby('stratum')['failure_type'].value_counts(normalize=True).unstack().fillna(0)
    stratum_groups.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title("Phase 2: Failure Taxonomy by Scenario Stratum")
    plt.ylabel("Proportion of Sample")
    plt.xlabel("Stratum")
    plt.legend(title="Failure Type\\n0: None, 1: Drift, 2: Switch, 3: Break, 4: Col.", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase2_failure_taxonomy.png"))
    plt.close()
    print("Saved Phase 2 Failure Taxonomy Plot.")
    
    # 2. Chain Length Fingerprint (Chain length vs failure probability)
    # We will approximate this by viewing the FP16 chain score as proxy for Chain Length / coherence required.
    # Grouped into bins for simplicity
    df['fp16_chain_bins'] = pd.cut(df['fp16_chain_score'], bins=[-1, 0.25, 0.5, 0.75, 1.0], labels=['0-1 Steps', '1-2 Steps', '2-3 Steps', '4+ Steps'])
    df['is_execution_failure'] = df['failure_type'].isin([1, 3]) # Types 1 & 3 are execution failures
    
    cl_prob = df.groupby('fp16_chain_bins')['is_execution_failure'].mean()
    
    plt.figure(figsize=(8, 6))
    plt.plot(cl_prob.index, cl_prob.values, marker='o', linewidth=2, color='red')
    plt.title("Chain Length Fingerprint\\nFailure Prob. vs. Coherent Steps Required")
    plt.xlabel("Required Coherent Steps (FP16 Chain Length Proxy)")
    plt.ylabel("Probability of Execution Failure (Type 1+3)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase2_chain_length_fingerprint.png"))
    plt.close()
    print("Saved Phase 2 Chain Length Fingerprint.")

if __name__ == "__main__':
    # Add dummy test data runner
    pass
