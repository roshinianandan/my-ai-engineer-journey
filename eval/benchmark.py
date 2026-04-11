import json
import os
import time
from datetime import datetime
import ollama
from eval.metrics import score_answer
from eval.judge import llm_judge
from config import MODEL

TEST_CASES_PATH = "./eval/test_cases.json"
REPORTS_DIR = "./eval/reports"


def load_test_cases(path: str = TEST_CASES_PATH) -> list:
    with open(path, "r") as f:
        return json.load(f)


def generate_answer(question: str) -> str:
    """Generate an answer using the main model."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Answer questions clearly and concisely."
            },
            {"role": "user", "content": question}
        ],
        stream=False,
        options={"temperature": 0.3}
    )
    return response["message"]["content"]


def run_benchmark(
    use_judge: bool = False,
    category_filter: str = None,
    save_report: bool = True
) -> dict:
    """
    Run all test cases and generate a full evaluation report.

    use_judge: also run LLM-as-judge scoring (slower but richer)
    category_filter: only run tests from a specific category
    save_report: save the full report to a JSON file
    """
    test_cases = load_test_cases()

    if category_filter:
        test_cases = [tc for tc in test_cases
                      if tc["category"] == category_filter]

    print(f"\n{'='*60}")
    print(f"  EVALUATION BENCHMARK")
    print(f"  Model: {MODEL}")
    print(f"  Test cases: {len(test_cases)}")
    if category_filter:
        print(f"  Category filter: {category_filter}")
    print(f"{'='*60}\n")

    results = []
    total_start = time.time()

    for i, tc in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {tc['id']}: {tc['question'][:50]}...")

        # Generate answer
        start = time.time()
        generated = generate_answer(tc["question"])
        gen_time = round(time.time() - start, 2)

        # Score with metrics
        scores = score_answer(generated, tc["expected_answer"])

        result = {
            "id": tc["id"],
            "category": tc["category"],
            "difficulty": tc["difficulty"],
            "question": tc["question"],
            "expected": tc["expected_answer"],
            "generated": generated,
            "generation_time": gen_time,
            "scores": scores
        }

        # Optional LLM judge
        if use_judge:
            print(f"  Running LLM judge...")
            judge_scores = llm_judge(
                tc["question"],
                tc["expected_answer"],
                generated
            )
            result["judge"] = judge_scores

        results.append(result)

        # Print result
        g = scores["grade"]
        c = scores["combined"]
        s = scores["semantic"]
        print(f"  Grade: {g} | Combined: {c:.3f} | "
              f"Semantic: {s:.3f} | Time: {gen_time}s\n")

    total_time = round(time.time() - total_start, 2)

    # Aggregate stats
    combined_scores = [r["scores"]["combined"] for r in results]
    semantic_scores = [r["scores"]["semantic"] for r in results]
    grades = [r["scores"]["grade"] for r in results]

    summary = {
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(results),
        "total_time_seconds": total_time,
        "avg_combined_score": round(sum(combined_scores) / len(combined_scores), 4),
        "avg_semantic_score": round(sum(semantic_scores) / len(semantic_scores), 4),
        "grade_distribution": {
            g: grades.count(g) for g in ["A", "B", "C", "D", "F"]
        },
        "category_scores": _category_breakdown(results)
    }

    report = {"summary": summary, "results": results}

    # Print summary
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Total cases:      {summary['total_cases']}")
    print(f"  Total time:       {total_time}s")
    print(f"  Avg combined:     {summary['avg_combined_score']:.4f}")
    print(f"  Avg semantic:     {summary['avg_semantic_score']:.4f}")
    print(f"  Grade distribution: {summary['grade_distribution']}")
    print(f"\n  Category Breakdown:")
    for cat, score in summary["category_scores"].items():
        print(f"    {cat}: {score:.4f}")

    # Save report
    if save_report:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        filename = f"{REPORTS_DIR}/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved: {filename}")

    return report


def _category_breakdown(results: list) -> dict:
    """Calculate average combined score per category."""
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["scores"]["combined"])

    return {
        cat: round(sum(scores) / len(scores), 4)
        for cat, scores in categories.items()
    }


def compare_reports(report1_path: str, report2_path: str):
    """Compare two benchmark reports to detect regressions."""
    with open(report1_path) as f:
        r1 = json.load(f)
    with open(report2_path) as f:
        r2 = json.load(f)

    s1 = r1["summary"]["avg_combined_score"]
    s2 = r2["summary"]["avg_combined_score"]
    delta = round(s2 - s1, 4)
    direction = "IMPROVED" if delta > 0 else "REGRESSED" if delta < 0 else "UNCHANGED"

    print(f"\n{'='*60}")
    print(f"  REGRESSION TEST")
    print(f"{'='*60}")
    print(f"  Report 1 avg: {s1:.4f}")
    print(f"  Report 2 avg: {s2:.4f}")
    print(f"  Delta:        {delta:+.4f}  ({direction})")

    if delta < -0.05:
        print("\n  ⚠️  REGRESSION DETECTED — score dropped more than 0.05")
    elif delta > 0.05:
        print("\n  ✅  SIGNIFICANT IMPROVEMENT detected")
    else:
        print("\n  ✅  No significant regression detected")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM Evaluation Benchmark")
    parser.add_argument("--run",      action="store_true", help="Run full benchmark")
    parser.add_argument("--judge",    action="store_true", help="Include LLM-as-judge scoring")
    parser.add_argument("--category", type=str,            help="Filter by category")
    parser.add_argument("--compare",  nargs=2, metavar=("REPORT1", "REPORT2"),
                        help="Compare two report files for regression")
    args = parser.parse_args()

    if args.compare:
        compare_reports(args.compare[0], args.compare[1])
    elif args.run:
        run_benchmark(
            use_judge=args.judge,
            category_filter=args.category
        )
    else:
        parser.print_help()