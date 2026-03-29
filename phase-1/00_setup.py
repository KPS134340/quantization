"""
PHASE 1 SETUP — Install and verify all dependencies.
Run this ONCE before anything else.
"""

# ── Run this in your terminal first ──────────────────────────────────────────
# pip install transformers bitsandbytes datasets accelerate
#             pandas numpy scikit-learn seaborn matplotlib tqdm torch
# ─────────────────────────────────────────────────────────────────────────────

import torch

def check_environment():
    print("=" * 50)
    print("ENVIRONMENT CHECK")
    print("=" * 50)

    # 1. CUDA / GPU
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU found      : {gpu}")
        print(f"   VRAM           : {vram:.1f} GB")
        if vram < 16:
            print("⚠️  WARNING: less than 16 GB VRAM — INT4 loading may OOM")
    else:
        print("❌ No GPU detected. You MUST use a cloud GPU instance.")
        print("   Recommended: RunPod, Lambda Labs, or Google Colab Pro+")
        return False

    # 2. Core libraries
    try:
        import transformers; print(f"✅ transformers   : {transformers.__version__}")
        import bitsandbytes; print(f"✅ bitsandbytes   : {bitsandbytes.__version__}")
        import datasets;     print(f"✅ datasets        : {datasets.__version__}")
        import pandas;       print(f"✅ pandas          : {pandas.__version__}")
        import sklearn;      print(f"✅ scikit-learn    : {sklearn.__version__}")
        import seaborn;      print(f"✅ seaborn         : {seaborn.__version__}")
    except ImportError as e:
        print(f"❌ Missing library: {e}")
        print("   Run: pip install transformers bitsandbytes datasets "
              "accelerate pandas numpy scikit-learn seaborn matplotlib tqdm")
        return False

    print("\n✅ Environment looks good. Proceed to 01_load_datasets.py")
    return True


if __name__ == "__main__":
    check_environment()
