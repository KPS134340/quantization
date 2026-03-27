import argparse
from phase1_audit import run_phase1
from phase2_mechanistic import run_phase2
from analysis import analyze_phase1, analyze_phase2

def parse_args():
    parser = argparse.ArgumentParser(description="VPPQ Stage 1 Pipeline (QCMST)")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="HuggingFace Model ID (e.g., meta-llama/Meta-Llama-3-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.3)")
    parser.add_argument("--phase1", action="store_true", help="Run Phase 1: Output Divergence Audit")
    parser.add_argument("--phase2", action="store_true", help="Run Phase 2: Mechanistic Decomposition")
    parser.add_argument("--samples", type=int, default=50, help="Max samples per sub-dataset for Phase 1")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for output CSVs")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Directory for generated plots")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.phase1:
        print(f"========== Starting Phase 1 Divergence Audit for {args.model} ==========")
        run_phase1(model_id=args.model, sample_size=args.samples, output_dir=args.output_dir)
        import os
        results_file = os.path.join(args.output_dir, f"phase1_{args.model.replace('/', '_')}_results.csv")
        analyze_phase1(results_file, output_dir=args.plots_dir)
        
    if args.phase2:
        import os
        if not os.environ.get("GROQ_API_KEY"):
            print("\\n[ERROR] GROQ_API_KEY is not set. Phase 2 requires it for the Judge LLM.")
            return

        print(f"\\n========== Starting Phase 2 Mechanistic Decomposition for {args.model} ==========")
        run_phase2(model_id=args.model, output_dir=args.output_dir)
        results_file = os.path.join(args.output_dir, f"phase2_{args.model.replace('/', '_')}_evaluated.csv")
        analyze_phase2(results_file, output_dir=args.plots_dir)

if __name__ == "__main__":
    main()
