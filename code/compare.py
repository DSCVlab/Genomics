import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_METHODS = ["rf", "anova", "pca", "hybrid"]
FEATURE_METHOD_LABELS = {
    "rf": "Random Forest Importance",
    "anova": "ANOVA F-test",
    "pca": "PCA",
    "hybrid": "ANOVA + RF Hybrid",
}
RANDOM_SEEDS = [42, 123, 456, 789, 999]


def load_spark_report(results_dir: str, method: str):
    path = os.path.join(results_dir, method, "aggregated", "final_report.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_sklearn_report(results_dir: str, method: str):
    path = os.path.join(results_dir, method, "aggregated_sklearn", "final_report_sklearn.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def per_seed_metric_array(report, model_name, metric_name, seeds):
    values = []
    for seed in seeds:
        entry = report.get("per_seed", {}).get(f"seed_{seed}")
        if entry is None:
            values.append(np.nan)
            continue
        v = entry.get("metrics", {}).get(model_name, {}).get(metric_name)
        values.append(v if v is not None else np.nan)
    return values


def paired_significance(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    res = {"n_paired": int(len(a)), "mean_diff": None, "ttest_p": None, "wilcoxon_p": None, "note": None}
    if len(a) < 2:
        res["note"] = "Fewer than 2 paired seeds available - cannot test."
        return res
    res["mean_diff"] = float(np.mean(a - b))
    try:
        _, p = stats.ttest_rel(a, b)
        res["ttest_p"] = float(p)
    except Exception as e:
        res["note"] = f"t-test failed: {e}"
    if len(a) >= 5 and not np.all(a == b):
        try:
            _, p = stats.wilcoxon(a, b)
            res["wilcoxon_p"] = float(p)
        except Exception as e:
            res["note"] = ((res["note"] or "") + f" wilcoxon failed: {e}").strip()
    else:
        res["note"] = ((res["note"] or "") + " Wilcoxon skipped (need >=5 non-identical paired seeds).").strip()
    return res


def cross_method_table(reports: dict):
    rows = []
    for method, report in reports.items():
        if report is None:
            continue
        ms = {r["Metric"]: r for r in report["metric_summary"]}
        row = {"Method": FEATURE_METHOD_LABELS.get(method, method), "method_key": method,
               "n_seeds": report["n_seeds_succeeded"]}
        for metric in ["ROC-AUC", "PR-AUC", "F1-score", "Accuracy", "Sensitivity", "Specificity", "Precision"]:
            key = f"stacking_ensemble_{metric}"
            if key in ms:
                row[f"{metric}_mean"] = ms[key]["Mean"]
                row[f"{metric}_std"] = ms[key]["Std"]
        rows.append(row)
    return pd.DataFrame(rows)


def significance_block(reports: dict, seeds):
    out = {}
    if "rf" not in reports or reports["rf"] is None:
        return out
    for other in ["anova", "pca", "hybrid"]:
        if other not in reports or reports[other] is None:
            continue
        out[f"rf_vs_{other}"] = {}
        for metric in ["ROC-AUC", "F1-score", "PR-AUC"]:
            a = per_seed_metric_array(reports["rf"], "stacking_ensemble", metric, seeds)
            b = per_seed_metric_array(reports[other], "stacking_ensemble", metric, seeds)
            out[f"rf_vs_{other}"][metric] = paired_significance(a, b)
    return out


def _get_seed_wall_time_and_mem(report, seed):
    """Handles both engines' per-seed field layouts (spark has an extra
    'config' key that sklearn doesn't, but both share 'resource' and
    'subprocess_wall_time_sec')."""
    if report is None:
        return None, None
    entry = report.get("per_seed", {}).get(f"seed_{seed}")
    if entry is None:
        return None, None
    resource = entry.get("resource", {}) or {}
    wall = entry.get("subprocess_wall_time_sec")
    if wall is None:
        wall = resource.get("total_training_time_sec")
    mem = resource.get("peak_driver_memory_mb")
    return wall, mem


def engine_resource_table(spark_reports: dict, sklearn_reports: dict, seeds):
    rows = []
    for method in FEATURE_METHODS:
        sr = spark_reports.get(method)
        kr = sklearn_reports.get(method)
        for seed in seeds:
            s_wall, s_mem = _get_seed_wall_time_and_mem(sr, seed)
            k_wall, k_mem = _get_seed_wall_time_and_mem(kr, seed)
            rows.append({
                "method": method, "seed": seed,
                "spark_wall_time_sec": s_wall, "spark_peak_mb": s_mem,
                "sklearn_wall_time_sec": k_wall, "sklearn_peak_mb": k_mem,
            })
    return pd.DataFrame(rows)


def plot_engine_comparison(df: pd.DataFrame, output_dir: str):
    """Each of the 4 bar-groups (spark/sklearn x time/memory) is checked
    and coerced to numeric NaN INDEPENDENTLY, since a column can be
    missing for a given method/seed even when others are populated (e.g.
    a run that failed, or hasn't been executed for that engine yet)."""
    df = df.copy()
    for col in ["spark_wall_time_sec", "spark_peak_mb", "sklearn_wall_time_sec", "sklearn_peak_mb"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")


    if not df["sklearn_peak_mb"].notna().any():
        print("[note] No sklearn peak-memory data found under any method/seed - "
              "make sure `--engine sklearn` runs have completed and "
              "`--aggregate --feature-method <m>` was run for sklearn.")
    if not df["spark_peak_mb"].notna().any():
        print("[note] No spark peak-memory data found under any method/seed - "
              "make sure `--engine spark` runs have completed and "
              "`--aggregate --feature-method <m>` was run for spark.")

    methods = df["method"].unique()
    fig, axes = plt.subplots(len(methods), 2, figsize=(14, 5 * len(methods)))
    if len(methods) == 1:
        axes = axes.reshape(1, -1)

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        x = np.arange(len(sub))
        width = 0.35

        ax = axes[i, 0]
        has_spark_time = sub["spark_wall_time_sec"].notna().any()
        has_sklearn_time = sub["sklearn_wall_time_sec"].notna().any()
        if has_spark_time:
            ax.bar(x - width / 2, sub["spark_wall_time_sec"], width, label="Spark", color="#2E86AB")
        if has_sklearn_time:
            ax.bar(x + width / 2, sub["sklearn_wall_time_sec"], width, label="sklearn", color="#F18F01")
        ax.set_xticks(x); ax.set_xticklabels(sub["seed"])
        ax.set_xlabel("Seed"); ax.set_ylabel("Time (s)")
        ax.set_title(f"{FEATURE_METHOD_LABELS.get(method, method)} — Training Time")
        if has_spark_time or has_sklearn_time:
            ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[i, 1]
        has_spark_mem = sub["spark_peak_mb"].notna().any()
        has_sklearn_mem = sub["sklearn_peak_mb"].notna().any()
        if has_spark_mem:
            ax.bar(x - width / 2, sub["spark_peak_mb"], width, label="Spark", color="#2E86AB")
        if has_sklearn_mem:
            ax.bar(x + width / 2, sub["sklearn_peak_mb"], width, label="sklearn", color="#F18F01")
        ax.set_xticks(x); ax.set_xticklabels(sub["seed"])
        ax.set_xlabel("Seed"); ax.set_ylabel("Peak memory (MB)")
        title = f"{FEATURE_METHOD_LABELS.get(method, method)} — Peak Memory"
        if not (has_spark_mem or has_sklearn_mem):
            title += " (no data)"
        ax.set_title(title)
        if has_spark_mem or has_sklearn_mem:
            ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "engine_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Spark vs sklearn results across all feature methods (rf/anova/pca/hybrid)."
    )
    parser.add_argument("--results-dir", type=str, default="./Lung_1_results_stacking",
                        help="Output dir used by the pipeline for BOTH --engine spark and "
                             "--engine sklearn runs (they share the same --output-dir, just "
                             "different subfolders per method/seed).")
    parser.add_argument("--spark-dir", type=str, default=None,
                        help="Override: use a different dir than --results-dir for Spark reports only.")
    parser.add_argument("--sklearn-dir", type=str, default=None,
                        help="Override: use a different dir than --results-dir for sklearn reports only.")
    parser.add_argument("--output-dir", type=str, default="./comparison_results")
    parser.add_argument("--seeds", type=int, nargs="+", default=RANDOM_SEEDS)
    parser.add_argument("--methods", type=str, nargs="+", default=FEATURE_METHODS,
                        choices=FEATURE_METHODS,
                        help="Subset of methods to compare (default: all of rf/anova/pca/hybrid).")
    args = parser.parse_args()

    spark_dir = args.spark_dir or args.results_dir
    sklearn_dir = args.sklearn_dir or args.results_dir

    os.makedirs(args.output_dir, exist_ok=True)

    spark_reports = {m: load_spark_report(spark_dir, m) for m in args.methods}
    sklearn_reports = {m: load_sklearn_report(sklearn_dir, m) for m in args.methods}

    missing_spark = [m for m, r in spark_reports.items() if r is None]
    missing_sklearn = [m for m, r in sklearn_reports.items() if r is None]
    if missing_spark:
        print(f"[warn] No Spark aggregated report found for: {missing_spark} "
              f"(run: python3 <pipeline>.py --aggregate --feature-method <m>  "
              f"-- looks for {spark_dir}/<m>/aggregated/final_report.json)")
    if missing_sklearn:
        print(f"[warn] No sklearn aggregated report found for: {missing_sklearn} "
              f"(run: python3 <pipeline>.py --aggregate --feature-method <m>  "
              f"-- looks for {sklearn_dir}/<m>/aggregated_sklearn/final_report_sklearn.json)")

    lines = []
    lines.append("=" * 100)
    lines.append("CROSS-METHOD COMPARISON — SPARK ENGINE")
    lines.append("=" * 100)
    spark_table = cross_method_table(spark_reports)
    if len(spark_table):
        spark_table.to_csv(os.path.join(args.output_dir, "cross_method_spark.csv"), index=False)
        lines.append(spark_table.to_string(index=False))
    else:
        lines.append("(no Spark reports found)")

    lines.append("")
    lines.append("=" * 100)
    lines.append("CROSS-METHOD COMPARISON — SKLEARN ENGINE")
    lines.append("=" * 100)
    sklearn_table = cross_method_table(sklearn_reports)
    if len(sklearn_table):
        sklearn_table.to_csv(os.path.join(args.output_dir, "cross_method_sklearn.csv"), index=False)
        lines.append(sklearn_table.to_string(index=False))
    else:
        lines.append("(no sklearn reports found)")

    sig_spark = significance_block(spark_reports, args.seeds)
    with open(os.path.join(args.output_dir, "significance_spark.json"), "w") as f:
        json.dump(sig_spark, f, indent=2, default=str)
    lines.append("")
    lines.append("=" * 100)
    lines.append("SIGNIFICANCE TESTS (paired by seed, RF as reference) — SPARK ENGINE")
    lines.append("=" * 100)
    for pair_key, metrics_dict in sig_spark.items():
        lines.append(f"  {pair_key}:")
        for metric, res in metrics_dict.items():
            tp = f"{res['ttest_p']:.4f}" if res["ttest_p"] is not None else "n/a"
            wp = f"{res['wilcoxon_p']:.4f}" if res["wilcoxon_p"] is not None else "n/a"
            md = f"{res['mean_diff']:+.4f}" if res["mean_diff"] is not None else "n/a"
            lines.append(f"    {metric:<10} mean_diff={md:>10}  paired_t p={tp:>8}  wilcoxon p={wp:>8}  (n={res['n_paired']})")

    sig_sklearn = significance_block(sklearn_reports, args.seeds)
    with open(os.path.join(args.output_dir, "significance_sklearn.json"), "w") as f:
        json.dump(sig_sklearn, f, indent=2, default=str)
    lines.append("")
    lines.append("=" * 100)
    lines.append("SIGNIFICANCE TESTS (paired by seed, RF as reference) — SKLEARN ENGINE")
    lines.append("=" * 100)
    for pair_key, metrics_dict in sig_sklearn.items():
        lines.append(f"  {pair_key}:")
        for metric, res in metrics_dict.items():
            tp = f"{res['ttest_p']:.4f}" if res["ttest_p"] is not None else "n/a"
            wp = f"{res['wilcoxon_p']:.4f}" if res["wilcoxon_p"] is not None else "n/a"
            md = f"{res['mean_diff']:+.4f}" if res["mean_diff"] is not None else "n/a"
            lines.append(f"    {metric:<10} mean_diff={md:>10}  paired_t p={tp:>8}  wilcoxon p={wp:>8}  (n={res['n_paired']})")

    lines.append("")
    lines.append("  NOTE: n=5 seeds is a small sample; treat p-values as indicative, report")
    lines.append("  effect sizes (mean_diff) alongside them.")

    eng_df = engine_resource_table(spark_reports, sklearn_reports, args.seeds)
    eng_df.to_csv(os.path.join(args.output_dir, "engine_comparison.csv"), index=False)
    lines.append("")
    lines.append("=" * 100)
    lines.append("SPARK vs SKLEARN — TRAINING TIME & PEAK MEMORY (per method, per seed)")
    lines.append("=" * 100)
    lines.append(eng_df.to_string(index=False))

    if eng_df["spark_wall_time_sec"].notna().any() or eng_df["sklearn_wall_time_sec"].notna().any():
        plot_engine_comparison(eng_df, args.output_dir)
        lines.append("")
        lines.append(f"[plot saved] {os.path.join(args.output_dir, 'engine_comparison.png')}")

  
    summary_text = "\n".join(lines)
    print(summary_text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    print(f"\n✓ All outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

