"""
PHASE 1 STEP 2 — Load models in FP16, INT8, and INT4.

Run this BEFORE inference to confirm all three variants load correctly
on your GPU without running out of memory.

Models:
  - meta-llama/Meta-Llama-3-8B-Instruct
  - mistralai/Mistral-7B-Instruct-v0.3

You must have accepted the HuggingFace license agreement for Llama-3
and be logged in:  huggingface-cli login
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


# ── Model identifiers on HuggingFace Hub ─────────────────────────────────────
MODEL_IDS = {
    "llama3":   "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral":  "mistralai/Mistral-7B-Instruct-v0.3",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ════════════════════════════════════════════════════════════════════════════
# PRECISION CONFIGURATIONS
# ════════════════════════════════════════════════════════════════════════════

def get_fp16_config():
    """Full precision — baseline reference."""
    return {"torch_dtype": torch.float16, "device_map": "auto"}


def get_int8_config():
    """Mild compression via bitsandbytes INT8."""
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    return {
        "quantization_config": quantization_config,
        "device_map": "auto",
    }


def get_int4_config():
    """Aggressive compression — NF4 4-bit via bitsandbytes."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NF4 = Normal Float 4, best quality
        bnb_4bit_compute_dtype=torch.float16, # Upcasts during compute
        bnb_4bit_use_double_quant=True,      # Double quantization reduces memory further
    )
    return {
        "quantization_config": quantization_config,
        "device_map": "auto",
    }


PRECISION_CONFIGS = {
    "FP16": get_fp16_config,
    "INT8": get_int8_config,
    "INT4": get_int4_config,
}


# ════════════════════════════════════════════════════════════════════════════
# LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_key: str, precision: str):
    """
    Load a model at a given precision.

    Parameters
    ----------
    model_key : str   — "llama3" or "mistral"
    precision : str   — "FP16", "INT8", or "INT4"

    Returns
    -------
    (model, tokenizer)
    """
    model_id = MODEL_IDS[model_key]
    config   = PRECISION_CONFIGS[precision]()

    print(f"\nLoading {model_key} @ {precision} ...")
    print(f"  Model ID : {model_id}")

    # Tokenizer is precision-independent
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  # needed for batch padding

    # Model with the precision-specific config
    model = AutoModelForCausalLM.from_pretrained(model_id, **config)
    model.eval()  # disable dropout — we want deterministic outputs

    # Memory report
    if torch.cuda.is_available():
        used_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory used: {used_gb:.2f} GB")

    print(f"  ✅ {model_key} @ {precision} loaded successfully")
    return model, tokenizer


def unload_model(model):
    """
    Explicitly free GPU memory after you're done with a precision variant.
    Call this between loading INT8 and INT4 to avoid OOM.
    """
    del model
    torch.cuda.empty_cache()
    print("  🗑  Model unloaded and GPU cache cleared.")


# ════════════════════════════════════════════════════════════════════════════
# SMOKE TEST — verify each config loads without crashing
# Run this before the full inference loop
# ════════════════════════════════════════════════════════════════════════════

def smoke_test(model_key: str = "mistral"):
    """
    Load one model in all three precisions, run one forward pass each,
    then unload. Use this to confirm your GPU can handle all three configs.
    """
    test_prompt = "Is it morally acceptable to lie to protect someone's feelings? Answer yes or no."

    for precision in ["FP16", "INT8", "INT4"]:
        model, tokenizer = load_model_and_tokenizer(model_key, precision)

        inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,   # greedy decoding
                temperature=1.0,   # ignored under greedy, but set for clarity
            )
        answer = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
        print(f"  [{precision}] model answer: {repr(answer)}")

        unload_model(model)

    print("\n✅ Smoke test passed. Proceed to 03_run_inference.py")


if __name__ == "__main__":
    smoke_test(model_key="mistral")   # change to "llama3" if preferred
