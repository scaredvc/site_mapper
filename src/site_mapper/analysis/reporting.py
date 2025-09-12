"""
Minimal console reporting helpers used by test_models.
"""

from __future__ import annotations

from typing import Dict, Any


def print_overview(report: Dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"URL: {report['url']}")
    print(f"Models tested: {len(report['models_tested'])}")
    print(f"Links processed: {report['total_links']}")


def print_performance_table(performance: Dict[str, Dict[str, Any]]) -> None:
    print("\nPERFORMANCE COMPARISON:")
    print(f"{'Model':<20} {'Success Rate':<12} {'Total Cost':<12} {'Total Time':<12}")
    print(f"{'-'*70}")
    for model_id, perf in performance.items():
        success_rate = perf.get('success_rate', 0)
        total_cost = perf.get('total_cost', 0)
        total_time = perf.get('total_time', 0)
        print(f"{model_id:<20} {success_rate:.1%}        ${total_cost:.4f}      {total_time:.2f}s")


def print_distributions(results: Dict[str, Any]) -> None:
    print("\nCLASSIFICATION DISTRIBUTION:")
    for model_id, model_results in results.items():
        if not model_results:
            continue
        counts: Dict[str, int] = {}
        total = len(model_results)
        for r in model_results:
            if r.get('success', False):
                lbl = r.get('classification', 'unknown')
                counts[lbl] = counts.get(lbl, 0) + 1
        print(f"\n{model_id}:")
        for lbl, count in counts.items():
            pct = (count / total) * 100 if total else 0
            print(f"  {lbl}: {count} ({pct:.1f}%)")


