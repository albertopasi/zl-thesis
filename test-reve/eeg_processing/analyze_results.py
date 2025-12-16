"""
Analyze comprehensive evaluation results and print summary statistics.
"""

import json
import numpy as np
from pathlib import Path

def analyze_results(json_path="comprehensive_results.json"):
    """Load results and print formatted summary."""
    
    with open(json_path) as f:
        results = json.load(f)
    
    print("\n" + "=" * 90)
    print("COMPREHENSIVE WORKLOAD CLASSIFICATION EVALUATION - DETAILED SUMMARY")
    print("=" * 90)
    
    # Analyze each scenario
    scenario_summaries = {}
    
    for scenario_name, configs in results.items():
        accs = [m["accuracy"] for m in configs.values()]
        kappas = [m["cohen_kappa"] for m in configs.values()]
        f1s = [m["f1_weighted"] for m in configs.values()]
        
        scenario_summaries[scenario_name] = {
            "accuracy": {
                "mean": np.mean(accs),
                "std": np.std(accs),
                "min": np.min(accs),
                "max": np.max(accs),
            },
            "kappa": {
                "mean": np.mean(kappas),
                "std": np.std(kappas),
            },
            "f1": {
                "mean": np.mean(f1s),
                "std": np.std(f1s),
            },
            "best_config": max(configs.items(), key=lambda x: x[1]["accuracy"])[0],
            "best_acc": max(accs),
        }
    
    # Print per-scenario details
    for scenario_name, summary in scenario_summaries.items():
        display_name = scenario_name.upper().replace("_", " ")
        print(f"\n{display_name}")
        print(f"  Accuracy:  {summary['accuracy']['mean']:.3f} ± {summary['accuracy']['std']:.3f}  "
              f"(range: {summary['accuracy']['min']:.3f} - {summary['accuracy']['max']:.3f})")
        print(f"  Kappa:     {summary['kappa']['mean']:.3f} ± {summary['kappa']['std']:.3f}")
        print(f"  F1:        {summary['f1']['mean']:.3f} ± {summary['f1']['std']:.3f}")
        print(f"  Best:      {summary['best_config']} = {summary['best_acc']:.3f}")
    
    # Overall ranking
    print("\n" + "=" * 90)
    print("SCENARIO RANKING (by mean accuracy)")
    print("=" * 90)
    
    ranked = sorted(scenario_summaries.items(), key=lambda x: x[1]["accuracy"]["mean"], reverse=True)
    
    for rank, (scenario_name, summary) in enumerate(ranked, 1):
        display_name = scenario_name.replace("_", " ")
        print(f"{rank}. {display_name:40} {summary['accuracy']['mean']:.3f} ± {summary['accuracy']['std']:.3f}")
    
    # Key insights
    print("\n" + "=" * 90)
    print("KEY INSIGHTS")
    print("=" * 90)
    
    within_subject_mean = np.mean([
        scenario_summaries["subject_PD089_only"]["accuracy"]["mean"],
        scenario_summaries["subject_PD094_only"]["accuracy"]["mean"]
    ])
    cross_subject_mean = np.mean([
        scenario_summaries["cross_subject_PD089_to_PD094"]["accuracy"]["mean"],
        scenario_summaries["cross_subject_PD094_to_PD089"]["accuracy"]["mean"]
    ])
    combined_mean = scenario_summaries["combined_split"]["accuracy"]["mean"]
    
    print(f"\nWithin-subject (avg):     {within_subject_mean:.3f}")
    print(f"Cross-subject (avg):      {cross_subject_mean:.3f}")
    print(f"Combined split:           {combined_mean:.3f}")
    print(f"Random baseline:          0.250")
    
    print(f"\nWithin-subject advantage: {(within_subject_mean - cross_subject_mean) / cross_subject_mean * 100:.1f}% higher")
    print(f"Overall improvement:      {(within_subject_mean - 0.25) / 0.25 * 100:.1f}% above random")
    
    # Stability analysis
    print("\n" + "=" * 90)
    print("SEED STABILITY (lower std = more stable)")
    print("=" * 90)
    
    stability = sorted(
        scenario_summaries.items(),
        key=lambda x: x[1]["accuracy"]["std"]
    )
    
    for scenario_name, summary in stability:
        display_name = scenario_name.replace("_", " ")
        print(f"{display_name:40} ± {summary['accuracy']['std']:.3f}")
    
    print("\n" + "=" * 90)


if __name__ == "__main__":
    analyze_results()
