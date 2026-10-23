# Introduction
This code is an implementation of the paper, which is described in:
*Zhiqiang Li, Hongyu Zhang, Xiao-Yuan Jing, Wangyang Yu, Yueyue Liu. 
"Unsupervised Software Defect Prediction through Multi-view Clustering", 2025*. 
It has published in the Journal of *IEEE Transactions on Reliability*.

# Project Structure
This directory contains the following files:

* algorihtms: a file folder contains the implementations of MUSDP and some baseline methods

* data: there are 28 versions of 8 projects, and each project contains 65 software metrics

* utilities: a file folder contains some utility functions

* test: a file folder contains the main functions to run the MUSDP and the baseline methods

* result: a file folder contains some detailed experimental results

* supplementary: a file folder contains some supplementary results

# Usage
Please run demo_MUSDP to obtain the prediction resutls of our approaches.

Our running environment is Python 3.10, Ubuntu 22.04 operating system.

## Paired feature-importance comparison

From this `code` directory, run:

```powershell
python .\test\demo_MUSDP_importance_comparison.py
```

Dependencies: numpy, pandas, scipy, scikit-learn, and statsmodels.
The default experiment uses 100 bootstrap rounds per dataset, base seed
100 (derived separately for each round), and five permutations per feature.
Mean replacement removes a selected feature's variation using its mean in
the standardized evaluation sample. Both methods refit clustering with the
same seed as the baseline. Importance is baseline accuracy minus perturbed
accuracy, averaged over repeats for permutation.

A smaller initial run is:

```powershell
python .\test\demo_MUSDP_importance_comparison.py --datasets activemq-5.0.0 --reps 2 --permutation-repeats 2 --output .\result\MUSDP_comparison_small
```

The default output folder is `result/MUSDP_importance_comparison`. Choose a
new `--output` folder for subsequent experiments; existing folders are not
overwritten. Outputs include:

- `performance.csv`: baseline accuracy plus the original 13 measures, for
  both labeling strategies. Performance is shared by the two perturbation
  methods, so it is saved once per round and labeling strategy.
- `importance.csv`: feature names, view, seeds, baseline/perturbed accuracy,
  and importance for both perturbations and both labeling strategies.
- `selected_features.csv`, `skipped_rounds.csv`, and `settings.json`.
- `comparison_report/`: paired feature ranks, per-round and per-dataset
  Spearman/Kendall correlations and top-five Jaccard overlap, feature rank
  summaries, a dedicated Added_lines/Del_lines summary, and `report.md`.

The new runner labels each view independently and uses majority voting for
label1; label2 independently labels the smaller cluster as defective. This
corrects the original shared-label mutation, so results should be compared
between the new runner's two strategies, rather than directly against old
importance CSVs. Feature selection still occurs on each complete dataset
before bootstrap, matching the original experimental setup. Single-class
evaluation samples are logged and skipped because AUC is undefined.

To regenerate only the ranking reports from a completed experiment:

```powershell
python .\utilities\compare_importance.py .\result\MUSDP_importance_comparison\importance.csv .\result\MUSDP_importance_comparison\comparison_report
```

Focused validation checks:

```powershell
python .\test\test_importance_comparison.py
```


# NOTE 
The software is free for academic use only, and shall not be used, rewritten, or adapted as the basis of a commercial product without first obtaining permission from the authors. 
The authors make no representations about the suitability of this software for any purpose. It is provided "as is" without express or implied warranty.

