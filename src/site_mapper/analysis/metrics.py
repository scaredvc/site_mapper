"""
Metrics helpers for model test results: success rates, timing, cost, logprob stats, uncertainty.
"""

from __future__ import annotations

from typing import Dict, Any, List


def compute_model_performance(model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful_tests = sum(1 for r in model_results if r.get("success", False))
    total_cost = sum(r.get("cost", 0) for r in model_results)
    total_time = sum(r.get("processing_time", 0) for r in model_results)

    return {
        "total_tests": len(model_results),
        "successful_tests": successful_tests,
        "success_rate": successful_tests / len(model_results) if model_results else 0,
        "total_cost": total_cost,
        "total_time": total_time,
        "avg_time_per_test": total_time / len(model_results) if model_results else 0,
    }


def classification_distribution(model_results: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in model_results:
        if r.get("success", False):
            lbl = r.get("classification", "unknown")
            counts[lbl] = counts.get(lbl, 0) + 1
    return counts


