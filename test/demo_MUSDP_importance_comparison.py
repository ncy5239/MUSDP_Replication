"""Evaluate MUSDP and compare seeded permutation/mean-replacement importance."""
import argparse
import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from algorithms.coreg import MultiviewCoRegSpectralClustering
from algorithms.coreg.mean_replacement_importance import (
    mean_replacement_importance, permutation_importance,
)
from utilities.AutoSpearman import AutoSpearman
from utilities.bootstrapCV import outofsample_bootstrap
from utilities import performanceMeasure, rankMeasure
from utilities.compare_importance import compare_results

MEASURES = ["AUC", "G_mean", "precision", "recall", "pf", "F1", "MCC",
            "Popt", "recall_effort", "precision_effort", "F1_effort", "PMI", "IFA"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--datasets", nargs="*", help="Dataset stems, e.g. activemq-5.0.0")
    parser.add_argument("--output", type=Path,
                        default=CODE_DIR / "result" / "MUSDP_importance_comparison")
    args = parser.parse_args()
    if args.reps < 1 or args.permutation_repeats < 1 or args.seed < 0:
        parser.error("Repetitions must be positive and seed must be nonnegative")
    files = sorted((CODE_DIR / "data").glob("*.csv"))
    if args.datasets:
        requested = set(args.datasets)
        missing = requested - {p.stem for p in files}
        if missing:
            parser.error("Unknown datasets: " + ", ".join(sorted(missing)))
        files = [p for p in files if p.stem in requested]
    if not files:
        parser.error("No dataset CSV files found")
    # Avoid silently overwriting or mixing experiments.
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = vars(args).copy()
    metadata["output"] = str(args.output.resolve())
    metadata["datasets"] = [p.stem for p in files]
    metadata["label1"] = "majority vote across independently labeled views"
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    selected_rows, skipped = [], []
    for file in files:
        data = pd.read_csv(file)
        views = [AutoSpearman(data.iloc[:, a:b]) for a, b in [(0, 54), (54, 59), (59, 65)]]
        features = [(i + 1, name) for i, view in enumerate(views) for name in view.columns]
        selected_rows.extend({"dataset": file.stem, "view": v, "feature": f} for v, f in features)
        for round_id in range(args.reps):
            # Identical bootstrap seeds to the original demos; independent fit seeds.
            _, _, test, y, _, indices = outofsample_bootstrap(views + [data.iloc[:, -1]], round_id)
            if len(y) < 2 or len(np.unique(y)) < 2:
                skipped.append({"dataset": file.stem, "round": round_id,
                                "reason": "Fewer than two samples or actual classes"})
                continue
            seed = int(np.random.SeedSequence([args.seed, round_id]).generate_state(1)[0])
            test = [scale(x) for x in test]
            loc = data["CountLineCode"].iloc[indices].to_numpy()
            model = MultiviewCoRegSpectralClustering(n_clusters=2, v_lambda=1,
                                                     max_iter=10, random_state=seed)
            results = {
                "mean_replacement": mean_replacement_importance(model, test, y, seed),
                "permutation": permutation_importance(model, test, y, seed, args.permutation_repeats),
            }
            baseline = results["mean_replacement"]
            performance_rows, importance_rows = [], []
            for label_idx, prediction in enumerate(baseline["baseline_labels"]):
                values = list(performanceMeasure.get_measure(y, prediction))
                values += list(rankMeasure.rank_measure(prediction, loc, y))
                identity = {"dataset": file.stem, "round": round_id,
                            "labeling": "label" + str(label_idx + 1), "clustering_seed": seed}
                performance_rows.append(dict(identity, accuracy=baseline["baseline_accuracy"][label_idx],
                                             **dict(zip(MEASURES, values))))
                for strategy, result in results.items():
                    for feature_idx, (view, feature) in enumerate(features):
                        importance_rows.append(dict(identity, view=view, feature=feature, strategy=strategy,
                            baseline_accuracy=result["baseline_accuracy"][label_idx],
                            perturbed_accuracy=result["perturbed_accuracy"][feature_idx, label_idx],
                            importance=result["importance"][feature_idx, label_idx]))
            for name, rows in [("performance.csv", performance_rows), ("importance.csv", importance_rows)]:
                target = args.output / name
                pd.DataFrame(rows).to_csv(target, mode="a", header=not target.exists(), index=False)
            print(f"{file.stem}: round {round_id + 1}/{args.reps} saved", flush=True)
    pd.DataFrame(selected_rows).to_csv(args.output / "selected_features.csv", index=False)
    pd.DataFrame(skipped, columns=["dataset", "round", "reason"]).to_csv(args.output / "skipped_rounds.csv", index=False)
    if (args.output / "importance.csv").exists():
        compare_results(args.output / "importance.csv", args.output / "comparison_report")
        print("Results and comparison report saved to", args.output.resolve())
    else:
        print("No valid rounds; see skipped_rounds.csv")


if __name__ == "__main__":
    main()
