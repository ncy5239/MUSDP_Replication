"""Compare paired mean-replacement and permutation importance results."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def compare_results(importance_file, report_dir):
    """Save paired ranks, stability, feature summaries, and a readable report.

    Only selected features are ranked; unselected features are never given
    zero importance. Lower ranks mean higher importance. Ties use mean ranks.
    Top-five membership includes all ties at the fifth position.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(importance_file)
    keys = ["dataset", "round", "labeling", "view", "feature"]
    paired = data.pivot(index=keys, columns="strategy", values="importance").reset_index()
    strategies = ["permutation", "mean_replacement"]
    if not set(strategies).issubset(paired.columns):
        raise ValueError("Both perturbation strategies are required")
    if paired[strategies].isna().any().any():
        raise ValueError("Importance rows must be paired across strategies")
    groups = ["dataset", "round", "labeling"]
    for strategy in strategies:
        paired[strategy + "_rank"] = paired.groupby(groups)[strategy].rank(
            ascending=False, method="average")
        paired[strategy + "_top5"] = paired.groupby(groups)[strategy].rank(
            ascending=False, method="min") <= 5
    paired["rank_change"] = paired.mean_replacement_rank - paired.permutation_rank
    paired["absolute_rank_change"] = paired.rank_change.abs()
    paired.to_csv(report_dir / "paired_feature_ranks.csv", index=False)
    stability = []
    for key, frame in paired.groupby(groups):
        a, b = frame.permutation, frame.mean_replacement
        varying = len(frame) > 1 and a.nunique() > 1 and b.nunique() > 1
        top_a = set(zip(frame.loc[frame.permutation_top5, "view"],
                        frame.loc[frame.permutation_top5, "feature"]))
        top_b = set(zip(frame.loc[frame.mean_replacement_top5, "view"],
                        frame.loc[frame.mean_replacement_top5, "feature"]))
        stability.append(dict(zip(groups, key), n_features=len(frame),
                              spearman=spearmanr(a, b).statistic if varying else np.nan,
                              kendall=kendalltau(a, b).statistic if varying else np.nan,
                              top5_jaccard=len(top_a & top_b) / len(top_a | top_b),
                              mean_absolute_rank_change=frame.absolute_rank_change.mean()))
    stability = pd.DataFrame(stability)
    stability.to_csv(report_dir / "round_stability.csv", index=False)
    stability.groupby(["dataset", "labeling"]).mean(numeric_only=True).to_csv(
        report_dir / "dataset_stability.csv")
    summary = paired.groupby(["dataset", "labeling", "view", "feature"]).agg(
        selected_rounds=("round", "nunique"),
        permutation_mean_importance=("permutation", "mean"),
        mean_replacement_mean_importance=("mean_replacement", "mean"),
        permutation_median_rank=("permutation_rank", "median"),
        permutation_rank_q25=("permutation_rank", lambda x: x.quantile(.25)),
        permutation_rank_q75=("permutation_rank", lambda x: x.quantile(.75)),
        mean_replacement_median_rank=("mean_replacement_rank", "median"),
        mean_replacement_rank_q25=("mean_replacement_rank", lambda x: x.quantile(.25)),
        mean_replacement_rank_q75=("mean_replacement_rank", lambda x: x.quantile(.75)),
        mean_rank_change=("rank_change", "mean"),
        mean_absolute_rank_change=("absolute_rank_change", "mean"),
        permutation_top5_frequency=("permutation_top5", "mean"),
        mean_replacement_top5_frequency=("mean_replacement_top5", "mean"),
    ).reset_index()
    totals = paired.groupby(["dataset", "labeling"])["round"].nunique()
    summary["selection_frequency"] = [
        row.selected_rounds / totals.loc[(row.dataset, row.labeling)]
        for row in summary.itertuples()]
    summary.to_csv(report_dir / "feature_summary.csv", index=False)
    # Explicit rows also show when either requested feature was never selected.
    targets = pd.MultiIndex.from_tuples(
        [(d, l, f) for d, l in totals.index for f in ["Added_lines", "Del_lines"]],
        names=["dataset", "labeling", "feature"]).to_frame(index=False)
    targets = targets.merge(summary, on=["dataset", "labeling", "feature"], how="left")
    targets[["selected_rounds", "selection_frequency"]] = targets[
        ["selected_rounds", "selection_frequency"]].fillna(0)
    targets.to_csv(report_dir / "added_deleted_lines_summary.csv", index=False)
    text = (
        "# Perturbation ranking comparison\n\n"
        "Importance = baseline accuracy - perturbed accuracy. Positive rank change "
        "means a feature ranks lower under mean replacement. Ranks are paired within "
        "dataset, round, and labeling strategy. Ties use average ranks.\n\n"
        "See added_deleted_lines_summary.csv for Added_lines and Del_lines; "
        "feature_summary.csv includes all selected features and rank quartiles. "
        "Unselected target features have selection frequency zero and undefined ranks.\n\n"
        "AutoSpearman runs once per full dataset before bootstrap, as in the original "
        "demos; selection frequency will therefore normally be zero or one.\n\n"
        "Higher Spearman/Kendall correlation and top-five Jaccard overlap indicate "
        "greater stability. Correlations are undefined for constant rankings. "
        "Top-five sets include ties. No arbitrary stable/unstable cutoff is imposed.\n\n"
        "Mean stability across rounds (interpret per-dataset reports first):\n\n"
        + stability.groupby("labeling")[["spearman", "kendall", "top5_jaccard",
            "mean_absolute_rank_change"]].mean().to_string() + "\n"
    )
    (report_dir / "report.md").write_text(text, encoding="utf-8")
    return summary, stability


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("importance_file", type=Path)
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()
    compare_results(args.importance_file, args.report_dir)
