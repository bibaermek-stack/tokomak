"""Unified Master Training Pipeline for Tokamak AI Models.

Executes end-to-end training across all model tiers according to Google DeepMind
architectural principles:
  1. Google DeepMind TORAX Style Neural Transport Surrogate (train_transport_surrogate.py)
  2. DIII-D Experimental Neural Equilibrium Inversion (train_equilibrium_neural.py)
  3. J-TEXT Experimental Neural Disruption Predictor (train_jtext.py)
  4. Synthetic Disruption Risk Classifier (train_disruption.py)
  5. Google DeepMind TCV Magnetic RL Controller (tokamak.control.train with Asymmetric Critic & Plant Randomization)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def run_pipeline(ppo_steps: int = 20_000, skip_rl: bool = False, seed: int = 42):
    t_start = time.time()
    results = {}
    print("=" * 80)
    print("   STARTING UNIFIED TOKAMAK AI TRAINING PIPELINE (DEEPMIND ARCHITECTURE)   ")
    print("=" * 80)
    print(f"Random Seed: {seed} | PPO Steps: {ppo_steps} | RL Enabled: {not skip_rl}\n")

    # 1. TORAX Neural Transport Surrogate
    print("[1/5] TRAINING DEEPMIND TORAX NEURAL TRANSPORT SURROGATE...")
    try:
        from ml.train_transport_surrogate import train_surrogate
        rep1 = train_surrogate(n_samples=65_000, seed=seed)
        results["torax_surrogate"] = {
            "status": "success",
            "test_r2": rep1.get("test_r2"),
            "test_mae": rep1.get("test_mae_m2_s"),
            "latency_ms": rep1.get("inference_latency_ms")
        }
        print(">>> TORAX Surrogate: SUCCESS (Test R2: {:.4f})\n".format(rep1.get("test_r2", 0.0)))
    except Exception as e:
        print(f">>> TORAX Surrogate FAILED: {e}\n")
        results["torax_surrogate"] = {"status": "failed", "error": str(e)}

    # 2. DIII-D Neural Equilibrium Inversion
    print("[2/5] TRAINING DIII-D NEURAL EQUILIBRIUM INVERSION MODEL...")
    try:
        from ml.train_equilibrium_neural import train_neural_equilibrium
        rep2 = train_neural_equilibrium(max_iter=40)
        mean_r2 = rep2.get("overall", {}).get("mean_R2", 0.0)
        results["d3d_equilibrium"] = {
            "status": "success",
            "targets": rep2.get("targets", {}),
            "mean_r2": mean_r2,
            "samples": rep2.get("dataset", {}).get("total_samples")
        }
        print(f">>> DIII-D Equilibrium: SUCCESS (Mean R2: {mean_r2:.4f})\n")
    except Exception as e:
        print(f">>> DIII-D Equilibrium FAILED: {e}\n")
        results["d3d_equilibrium"] = {"status": "failed", "error": str(e)}

    # 3. J-TEXT Neural Disruption Predictor
    print("[3/5] TRAINING J-TEXT 1kHz NEURAL DISRUPTION PREDICTOR...")
    try:
        from ml.train_jtext import main as train_jtext_main
        rep3 = train_jtext_main()
        results["jtext_disruption"] = {
            "status": "success",
            "neural_auc": rep3.get("neural", {}).get("test_roc_auc") if rep3 else None,
            "classical_auc": rep3.get("test_roc_auc") if rep3 else None,
        }
        auc_val = rep3.get("neural", {}).get("test_roc_auc", 0.0) if rep3 else 0.0
        print(f">>> J-TEXT Disruption: SUCCESS (Neural Test AUC: {auc_val:.4f})\n")
    except Exception as e:
        print(f">>> J-TEXT Disruption FAILED: {e}\n")
        results["jtext_disruption"] = {"status": "failed", "error": str(e)}

    # 4. Synthetic Parametric Disruption Classifier
    print("[4/5] TRAINING PARAMETRIC SYNTHETIC DISRUPTION CLASSIFIER...")
    try:
        from ml.train_disruption import main as train_syn_main
        rep4 = train_syn_main()
        results["synthetic_disruption"] = {
            "status": "success",
            "accuracy": rep4.get("test_accuracy") if rep4 else None
        }
        print(">>> Synthetic Disruption: SUCCESS\n")
    except Exception as e:
        print(f">>> Synthetic Disruption FAILED: {e}\n")
        results["synthetic_disruption"] = {"status": "failed", "error": str(e)}

    # 5. DeepMind TCV Closed-Loop RL Controller
    if not skip_rl and ppo_steps > 0:
        print(f"[5/5] TRAINING DEEPMIND TCV RL CONTROLLER ({ppo_steps} PPO Steps)...")
        try:
            from tokamak.control.train import train as train_rl
            out_policy = str(backend_dir / "tokamak" / "control" / "policy_pipeline.npz")
            rep5 = train_rl(
                ppo_steps=ppo_steps,
                bc_episodes=15,
                rollout=1024,
                episode_steps=300,
                seed=seed,
                asymmetric=True,
                randomise_plant=True,
                out=out_policy
            )
            hist = rep5.get("history", {})
            final_eval = hist.get("final", {})
            results["rl_controller"] = {
                "status": "success",
                "final_return": final_eval.get("return"),
                "survival": final_eval.get("survival"),
                "pos_error_mm": final_eval.get("rms_position_error_mm"),
                "shape_error_mm": final_eval.get("rms_shape_error_mm")
            }
            print(f">>> DeepMind TCV Controller: SUCCESS (Survival: {final_eval.get('survival')}, Pos Error: {final_eval.get('rms_position_error_mm'):.1f} mm)\n")
        except Exception as e:
            print(f">>> DeepMind TCV Controller FAILED: {e}\n")
            results["rl_controller"] = {"status": "failed", "error": str(e)}
    else:
        print("[5/5] SKIPPING RL CONTROLLER (already trained or --skip-rl specified).\n")
        results["rl_controller"] = {"status": "skipped"}

    elapsed = time.time() - t_start
    summary_path = backend_dir / "ml" / "models" / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_elapsed_seconds": elapsed,
        "ppo_steps": ppo_steps,
        "results": results
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("=" * 80)
    print(f" ALL MODEL TRAININGS COMPLETED IN {elapsed:.1f} SECONDS! ")
    print(f" Summary saved to: {summary_path}")
    print("=" * 80)
    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified DeepMind-style Tokamak AI Training Pipeline")
    parser.add_argument("--ppo-steps", type=int, default=15_000, help="PPO steps for RL magnetic controller")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL training and only train neural predictors/surrogates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_pipeline(ppo_steps=args.ppo_steps, skip_rl=args.skip_rl, seed=args.seed)
