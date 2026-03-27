import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional
import os

class InferenceEngine:
    def __init__(self, model_id: str, precision: str = "fp16", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the model and tokenizer.
        precision can be 'fp16', 'int8', or 'int4'.
        """
        self.model_id = model_id
        self.precision = precision
        self.device = device
        
        print(f"Loading {model_id} in {precision} mode...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        kwargs = {
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
        }
        
        if precision == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif precision == "int8":
            kwargs["load_in_8bit"] = True
        elif precision == "int4":
            kwargs["load_in_4bit"] = True
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                print("Make sure `bitsandbytes` is installed for 4-bit quantization.")
                
        # Handle CPU only edge-cases
        if not torch.cuda.is_available() and precision in ["int8", "int4"]:
            print("WARNING: Qunatization requested but no GPU found. This may fail.")
            
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        print(f"Model loaded successfully.")
        
    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """
        Generates text using Greedy Decoding (temperature=0.0).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,      # Greedy decoding
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # Decode only the newly generated tokens
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

if __name__ == '__main__':
    # Simple test (might take a while or fail if no GPU / OOM)
    # Use a tiny model like gpt2 for a quick test if no GPU is present
    try:
        engine = InferenceEngine("gpt2", precision="fp16")
        print("Generated:", engine.generate("Is lying ever morally acceptable?"))
    except Exception as e:
        print("Test skipped or failed:", e)
