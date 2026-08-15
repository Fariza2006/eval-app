"""
metrics.py
-----------
Checkpoint 3: İzlənilən metriklər (accuracy/pass-rate, orta latency, orta token xərci)

evaluate.py-dəki run_evaluation() nəticələrini alıb, aqreqasiya edilmiş metrikləri
hesablayır. Token xərci llm_client.py-dəki _usage_log-dan oxunur.
"""

from llm_client import get_usage_log, reset_usage_log


def compute_metrics(results: list[dict]) -> dict:
    """
    run_evaluation()-dan gələn nəticələr siyahısını alıb, aqreqasiya edilmiş
    metrikləri hesablayır.

    Return:
        {
            "total": int,
            "correct": int,
            "incorrect": int,
            "unknown": int,             # judge etibarlı JSON qaytara bilmədiyi hallar
            "accuracy": float,           # correct / (correct + incorrect), unknown xaric
            "avg_latency_seconds": float,
            "avg_input_tokens": float,
            "avg_output_tokens": float,
            "avg_total_tokens": float,
            "avg_cost_usd": float,
            "by_category": dict,         # hər kateqoriya üzrə ayrı accuracy
        }
    """
    total = len(results)
    correct = sum(1 for r in results if r["correct"] is True)
    incorrect = sum(1 for r in results if r["correct"] is False)
    unknown = sum(1 for r in results if r["correct"] is None)

    scored = correct + incorrect  # accuracy hesablamasında "unknown" xaric edilir
    accuracy = round(correct / scored, 3) if scored > 0 else 0.0

    latencies = [r["latency_seconds"] for r in results if "latency_seconds" in r]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    # Token/xərc məlumatı llm_client-in usage log-undan gəlir
    usage_entries = get_usage_log()
    if usage_entries:
        avg_input = sum(u.get("prompt_tokens", 0) for u in usage_entries) / len(usage_entries)
        avg_output = sum(u.get("completion_tokens", 0) for u in usage_entries) / len(usage_entries)
        avg_total = sum(u.get("total_tokens", 0) for u in usage_entries) / len(usage_entries)
        avg_cost = sum(u.get("estimated_cost", 0) for u in usage_entries) / len(usage_entries)
    else:
        avg_input = avg_output = avg_total = avg_cost = 0.0

    # Kateqoriya üzrə ayrı accuracy (hansı sual tipi daha çox uğursuz olur?)
    by_category = {}
    categories = set(r["category"] for r in results)
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_correct = sum(1 for r in cat_results if r["correct"] is True)
        cat_scored = sum(1 for r in cat_results if r["correct"] is not None)
        by_category[cat] = {
            "total": len(cat_results),
            "correct": cat_correct,
            "accuracy": round(cat_correct / cat_scored, 3) if cat_scored > 0 else 0.0,
        }

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "unknown": unknown,
        "accuracy": accuracy,
        "avg_latency_seconds": avg_latency,
        "avg_input_tokens": round(avg_input, 1),
        "avg_output_tokens": round(avg_output, 1),
        "avg_total_tokens": round(avg_total, 1),
        "avg_cost_usd": round(avg_cost, 6),
        "by_category": by_category,
    }


def print_metrics_report(metrics: dict):
    """Metrikləri oxunaqlı formatda çap edir."""
    print("\n" + "=" * 50)
    print("METRİKLƏR HESABATI")
    print("=" * 50)
    print(f"Ümumi sual sayı:      {metrics['total']}")
    print(f"Düzgün:                {metrics['correct']}")
    print(f"Səhv:                  {metrics['incorrect']}")
    print(f"Naməlum (judge xətası): {metrics['unknown']}")
    print(f"Accuracy (pass-rate):  {metrics['accuracy'] * 100:.1f}%")
    print(f"Orta cavab müddəti:    {metrics['avg_latency_seconds']} saniyə")
    print(f"Orta giriş tokeni:     {metrics['avg_input_tokens']}")
    print(f"Orta çıxış tokeni:     {metrics['avg_output_tokens']}")
    print(f"Orta cəmi token:       {metrics['avg_total_tokens']}")
    print(f"Orta xərc (sorğu üçün): ${metrics['avg_cost_usd']}")
    print("\nKateqoriya üzrə accuracy:")
    for cat, stats in metrics["by_category"].items():
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({stats['accuracy'] * 100:.1f}%)")


if __name__ == "__main__":
    from test_set import DEV_SET
    from evaluate import run_evaluation

    reset_usage_log()  # təmiz başlanğıc - yalnız bu qiymətləndirmənin token istifadəsini ölçmək üçün

    print("=== CHECKPOINT 3: METRİKLƏRİN TOPLANMASI (DEV_SET üzərində) ===\n")
    results = run_evaluation(DEV_SET, verbose=True)

    metrics = compute_metrics(results)
    print_metrics_report(metrics)
