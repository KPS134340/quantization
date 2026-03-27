import os
from datasets import load_dataset
import pandas as pd
import random
from typing import Dict, List, Tuple

def load_ethics_dataset(split='test', num_samples=None):
    """Loads the Hendrycks ETHICS dataset."""
    print("Loading ETHICS dataset...")
    categories = ['commonsense', 'deontology', 'justice', 'utilitarianism', 'virtue']
    data = []
    
    for cat in categories:
        ds = load_dataset('hendrycks/ethics', cat, split=split, trust_remote_code=True)
        if num_samples is not None:
            # Safely sample
            n = min(len(ds), num_samples // len(categories))
            ds = ds.select(range(n))
            
        for row in ds:
            # The structure varies slightly per category, we standardise it
            text = row.get('text') or row.get('scenario') or ""
            label = row.get('label')
            data.append({
                'dataset': 'ethics',
                'category': cat,
                'text': text,
                'label': label,
                'raw_row': row
            })
    return pd.DataFrame(data)

def load_morebench(split='train', num_samples=None):
    """Loads morebench/morebench."""
    print("Loading MoreBench...")
    # MoreBench might not have standard test splits, using train for loading
    try:
        ds = load_dataset("morebench/morebench", split=split, trust_remote_code=True)
        if num_samples is not None:
            n = min(len(ds), num_samples)
            ds = ds.select(range(n))
            
        data = []
        for row in ds:
            # Assumed structure based on standard benchmarks
            text = row.get('question', row.get('text', ''))
            label = row.get('answer', row.get('label', ''))
            data.append({
                'dataset': 'morebench',
                'category': 'mixed', 
                'text': text,
                'label': label,
                'raw_row': row
            })
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Failed to load MoreBench: {e}")
        return pd.DataFrame()

def load_moral_stories(split='train', num_samples=None):
    """Loads Moral Stories dataset."""
    print("Loading Moral Stories...")
    try:
        ds = load_dataset("demelin/moral_stories", split=split, trust_remote_code=True)
        if num_samples is not None:
            n = min(len(ds), num_samples)
            ds = ds.select(range(n))
            
        data = []
        for row in ds:
            # Moral stories has situation, intention, moral_action, immoral_action
            situation = row.get('situation', '')
            intention = row.get('intention', '')
            action = row.get('moral_action', '')
            
            text = f"Situation: {situation}\\nIntention: {intention}\\nAction: {action}"
            data.append({
                'dataset': 'moral_stories',
                'category': 'consequentialist',
                'text': text,
                'label': 1, # Acceptable
                'raw_row': row
            })
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Failed to load Moral Stories: {e}")
        return pd.DataFrame()

def load_valuebench(split='train', num_samples=None):
    """Loads Value4AI/ValueBench dataset."""
    print("Loading ValueBench...")
    try:
        ds = load_dataset("Value4AI/ValueBench", split=split, trust_remote_code=True)
        if num_samples is not None:
            n = min(len(ds), num_samples)
            ds = ds.select(range(n))
            
        data = []
        for row in ds:
            text = row.get('prompt', row.get('text', ''))
            label = row.get('label', '')
            category = row.get('value_category', 'virtue')
            data.append({
                'dataset': 'valuebench',
                'category': category,
                'text': text,
                'label': label,
                'raw_row': row
            })
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Failed to load ValueBench: {e}")
        return pd.DataFrame()

def get_phase1_dataset(max_samples_per_dataset=50):
    """Compile a dataset for Phase 1 standard output divergence."""
    dfs = []
    
    df_ethics = load_ethics_dataset(num_samples=max_samples_per_dataset)
    if not df_ethics.empty: dfs.append(df_ethics)
        
    df_mb = load_morebench(num_samples=max_samples_per_dataset)
    if not df_mb.empty: dfs.append(df_mb)
        
    df_ms = load_moral_stories(num_samples=max_samples_per_dataset)
    if not df_ms.empty: dfs.append(df_ms)
        
    df_vb = load_valuebench(num_samples=max_samples_per_dataset)
    if not df_vb.empty: dfs.append(df_vb)
        
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def get_phase2_stratified_dataset():
    """
    Construct 400-scenario stratified sample for Phase 2:
    Stratum A: 150 temporal consequentialist (ETHICS Utilitarianism + Moral Stories)
    Stratum B: 100 deontological (ETHICS Deontology)
    Stratum C: 100 virtue/commonsense (ETHICS Commonsense/Virtue + ValueBench)
    Stratum D: 50 mixed (MoreBench)
    """
    print("Constructing 400-scenario stratified sample for Phase 2...")
    strata = []
    
    # Stratum A: Consequentialist (n=150)
    df_ethics_util = load_ethics_dataset(num_samples=75) # will mix all 5 cats, filter later
    df_ethics_util = df_ethics_util[df_ethics_util['category'] == 'utilitarianism'].head(75)
    df_ms = load_moral_stories(num_samples=150)
    stratum_a = pd.concat([df_ethics_util, df_ms.head(150 - len(df_ethics_util))], ignore_index=True)
    stratum_a['stratum'] = 'A_consequentialist'
    strata.append(stratum_a)
    
    # Stratum B: Deontological (n=100)
    df_ethics_deon = load_ethics_dataset(num_samples=200)
    stratum_b = df_ethics_deon[df_ethics_deon['category'] == 'deontology'].head(100)
    stratum_b['stratum'] = 'B_deontological'
    strata.append(stratum_b)
    
    # Stratum C: Virtue/Commonsense (n=100)
    df_ethics_vc = load_ethics_dataset(num_samples=200)
    stratum_c = df_ethics_vc[df_ethics_vc['category'].isin(['virtue', 'commonsense'])].head(50)
    df_vb = load_valuebench(num_samples=100)
    stratum_c = pd.concat([stratum_c, df_vb.head(100 - len(stratum_c))], ignore_index=True)
    stratum_c['stratum'] = 'C_virtue_commonsense'
    strata.append(stratum_c)
    
    # Stratum D: Mixed (n=50)
    stratum_d = load_morebench(num_samples=50)
    stratum_d['stratum'] = 'D_mixed'
    strata.append(stratum_d)
    
    final_df = pd.concat(strata, ignore_index=True)
    print(f"Generated Stratified Dataset Phase 2: {len(final_df)} samples")
    return final_df

if __name__ == '__main__':
    # Test data loading script
    p1 = get_phase1_dataset(max_samples_per_dataset=5)
    print("Phase 1 Preview:", len(p1))
    
    p2 = get_phase2_stratified_dataset()
    print("Phase 2 Previes:", p2['stratum'].value_counts())
