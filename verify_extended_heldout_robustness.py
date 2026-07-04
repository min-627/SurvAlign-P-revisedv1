import os
import argparse
import subprocess
import csv
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Extended Held-out Robustness Evaluation")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated list of seeds")
    parser.add_argument("--mode", required=True, help="Gate mode, e.g., proposed_gate")
    parser.add_argument("--map_type", required=True, help="Map type, e.g., survival")
    parser.add_argument("--dataset_type", default="librispeech")
    parser.add_argument("--dataset_name", default="dev-clean")
    parser.add_argument("--attacks", required=True, help="Comma-separated list of attacks")
    parser.add_argument("--run_prefix", default="ext_heldout", help="Prefix for run_id")
    
    # External commands
    parser.add_argument("--facodec_command", default="")
    parser.add_argument("--clearervoice_command", default="")
    parser.add_argument("--encodec_command", default="")
    parser.add_argument("--dac_command", default="")
    parser.add_argument("--vocos_command", default="")
    
    return parser.parse_args()

def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    requested_attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    
    # Filter attacks based on available commands
    valid_attacks = []
    for atk in requested_attacks:
        if atk == "facodec" and not args.facodec_command.strip():
            print(f"[SKIP] {atk}: Command wrapper not provided.")
            continue
        if atk == "clearervoice" and not args.clearervoice_command.strip():
            print(f"[SKIP] {atk}: Command wrapper not provided.")
            continue
        if atk == "encodec" and not args.encodec_command.strip():
            print(f"[SKIP] {atk}: Command wrapper not provided.")
            continue
        if atk == "dac" and not args.dac_command.strip():
            print(f"[SKIP] {atk}: Command wrapper not provided.")
            continue
        if atk == "vocos" and not args.vocos_command.strip():
            print(f"[SKIP] {atk}: Command wrapper not provided.")
            continue
        valid_attacks.append(atk)
        
    if not valid_attacks:
        print("No valid attacks to run after filtering missing wrappers. Exiting.")
        return

    print(f"Running evaluation for attacks: {', '.join(valid_attacks)}")
    
    run_ids = []
    
    for s in seeds:
        print(f"\n[Running Extended Eval - Seed {s}]")
        checkpoint_path = f"checkpoints/best_{args.dataset_type}_{args.mode}_{args.map_type}_seed{s}.pth"
        if not os.path.exists(checkpoint_path):
            print(f"[WARNING] Checkpoint {checkpoint_path} not found! Please ensure Phase 2 training is completed for seed {s}.")
            continue
            
        run_id = f"{args.run_prefix}_seed_{s}"
        run_ids.append(run_id)
        
        cmd = [
            "python", "phase2_training.py",
            "--mode", args.mode,
            "--map_type", args.map_type,
            "--dataset_type", args.dataset_type,
            "--dataset_name", args.dataset_name,
            "--test_only",
            "--load_weight", checkpoint_path,
            "--test_attacks", ",".join(valid_attacks),
            "--run_id", run_id
        ]
        
        if args.facodec_command.strip():
            cmd.extend(["--facodec_command", args.facodec_command.strip()])
        if args.clearervoice_command.strip():
            cmd.extend(["--clearervoice_command", args.clearervoice_command.strip()])
        if args.encodec_command.strip():
            cmd.extend(["--encodec_command", args.encodec_command.strip()])
        if args.dac_command.strip():
            cmd.extend(["--dac_command", args.dac_command.strip()])
        if args.vocos_command.strip():
            cmd.extend(["--vocos_command", args.vocos_command.strip()])
            
        subprocess.run(cmd, check=True)
        
    if not run_ids:
        print("\nNo seeds were successfully evaluated. Please check if checkpoints exist.")
        return
        
    print("\nAll seeds evaluated. Aggregating extended robustness results...")
    
    long_path = "results/phase2/phase2_results_long.csv"
    if not os.path.exists(long_path):
        print(f"Cannot find results file: {long_path}")
        return
        
    print("\n--- EXTENDED ROBUSTNESS SUMMARY ---")
    
    results = {}
    with open(long_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["run_id"] in run_ids and row["attack"] in valid_attacks:
                atk = row["attack"]
                sys = row["system"]
                acc = float(row["bit_accuracy"])
                exact = float(row["exact_message_accuracy"])
                
                if atk not in results:
                    results[atk] = {"baseline_acc": [], "method_acc": [], "baseline_exact": [], "method_exact": []}
                    
                if sys == "baseline":
                    results[atk]["baseline_acc"].append(acc)
                    results[atk]["baseline_exact"].append(exact)
                elif sys == "method":
                    results[atk]["method_acc"].append(acc)
                    results[atk]["method_exact"].append(exact)
                    
    for atk in valid_attacks:
        if atk not in results:
            print(f"Attack: {atk} | No data found in CSV.")
            continue
            
        b_acc = results[atk]["baseline_acc"]
        m_acc = results[atk]["method_acc"]
        b_ex = results[atk]["baseline_exact"]
        m_ex = results[atk]["method_exact"]
        
        b_acc_mean, b_acc_std = np.mean(b_acc)*100, np.std(b_acc)*100 if b_acc else (0,0)
        m_acc_mean, m_acc_std = np.mean(m_acc)*100, np.std(m_acc)*100 if m_acc else (0,0)
        b_ex_mean, b_ex_std = np.mean(b_ex)*100, np.std(b_ex)*100 if b_ex else (0,0)
        m_ex_mean, m_ex_std = np.mean(m_ex)*100, np.std(m_ex)*100 if m_ex else (0,0)
        
        print(f"\nAttack: {atk}")
        print(f"  Bit Accuracy   | Baseline: {b_acc_mean:6.2f}% +- {b_acc_std:5.2f}%  | Proposed: {m_acc_mean:6.2f}% +- {m_acc_std:5.2f}%")
        print(f"  Exact Match    | Baseline: {b_ex_mean:6.2f}% +- {b_ex_std:5.2f}%  | Proposed: {m_ex_mean:6.2f}% +- {m_ex_std:5.2f}%")

if __name__ == "__main__":
    main()
