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


## CLA and CLAMI baselines

Both demo files expose
`run_(X, save_path, project_name, model_name, randseed, percentile=50)`,
following the per-round structure in `demo_MUSDP.py`. Each function performs
bootstrap sampling, prediction, all 13 evaluations, and CSV saving under
`save_path/model_name/project_name.csv` (one appended row per round).
`X` is a DataFrame with metrics and the true defect label in its final column.
The functions also return the saved measures. Algorithm-only prediction
functions remain in `algorithms/CLA.py`.

`algorithms/CLA.py` exposes `CLA(metrics, percentile=50)` and
`CLAMI(metrics, percentile=50, classifier=None, random_state=0)`.
Pass only numeric metric columns (no defect labels). Both return binary
predictions for every input row, with 1 meaning buggy. They share clustering
and labeling code: count metrics strictly above their per-column percentile,
then label clusters above the median **distinct** count as buggy. An odd
middle cluster is assigned clean; one-cluster inputs receive all-clean CLA
labels.

CLAMI tries each distinct metric violation score in ascending order, selects
exactly the metrics with that score, and removes rows with any violation.
The first selection retaining both pseudo-labeled classes trains a classifier
and predicts every original row. Its default is standardized scikit-learn
logistic regression, following the paper's learner family, rather than an exact
reproduction of Weka's implementation. A custom sklearn classifier is cloned.
If no selection retains both classes, it raises `CLAMISelectionError`.

Run from the `code` directory (scripts also work from other directories):

```powershell
python .\test\demo_CLA.py
python .\test\demo_CLAMI.py
```

Each demo evaluates all datasets for 100 bootstrap rounds by default.
As in `demo_SC.py`, each method operates on the unlabeled out-of-bootstrap
evaluation sample itself; actual labels are used only for evaluation. The
13 output measures and their ordering match the existing demos. Results go
to `result/CLA` or `result/CLAMI`. Like `demo_MUSDP.py`, edit `save_path`,
`Reps`, and `percentile` in each script's `__main__` configuration block.
Results append to existing CSV files; use a fresh `save_path` for a separate
experiment. Unusable CLAMI selections and single-class evaluation samples
emit warnings and are skipped. Dependencies: numpy, pandas, scipy, scikit-learn.

## TCL and TCLP baselines

`algorithms/TCL.py` provides `TCL(metrics, return_details=False)` and
`TCLP(metrics, random_state=1, n_estimators=500, cv_folds=10,
tune_mtry=True, n_jobs=1, return_details=False)`. Input must contain at least
two modules and one raw, finite, nonnegative metric column. Do not include
true defect labels or standardize before threshold derivation. Outputs are
binary predictions for all original modules (1=faulty). With `return_details`,
each function returns `(predictions, details)` for inspection.

TCL follows equations (1)-(4) and the labeling rule in Kumar, Chaturvedi,
and Kailasam, *An Unsupervised Software Fault Prediction Approach Using
Threshold Derivation*, IEEE Transactions on Reliability 71(2), 2022:

- Transform metrics using `log1p`, then calculate mean plus sample SD (`ddof=1`).
- Use `exp(mean + SD)` as printed in equation (4), without subtracting one.
- Count raw metric values strictly above their thresholds. Label modules
  faulty when the count exceeds `floor(max_observed_count / 2)`.

TCLP reuses TCL's threshold and labeling code. It ranks metrics by MVR and
accumulates complete score groups until at least `max(1, ceil(log2(m)))`
metrics remain. It ranks modules by IVR on those metrics, starting at 70%
retention, then trying 80%, 90%, and 99%. Boundary ties are all included,
so the actual selection can exceed the requested fraction. A usable selection
must retain at least two modules in each class, as required by ROSE. If all
retention levels fail, it adds the next MVR group and repeats; otherwise it
raises `TCLPSelectionError`. These tie and expansion rules make the paper's
partially specified selection procedure deterministic.

Balancing implements numeric ROSE Gaussian smoothed bootstrap in Python,
using class-specific sample SD and normal-reference diagonal bandwidths:
`sd * (4 / ((p+2)*n_class))**(1/(p+4))`. As in ROSE's default configuration,
the generated sample has size `n` with class probability 0.5, so class balance
is approximate. Class allocation is redrawn if a generated sample contains
only one class. Synthetic values are continuous and are not clipped or rounded.
This follows the numeric kernel method described by
[the ROSE package](https://cran.r-universe.dev/ROSE/doc/manual.html);
it is not a bit-for-bit reproduction of the R package's random sequence.

The default random forest has 500 trees. `mtry` candidates `1..p` are evaluated
by stratified cross-validation accuracy against the selected pseudo labels;
ROSE is fitted independently within each training fold. Prefer smaller `mtry`
on tied accuracy. Use up to ten folds, reducing the fold count for small classes.
If any fold would have fewer than two modules in a class, tuning is skipped
and `max(1, floor(sqrt(p)))` is used. `details` and demo audit files explicitly
record this fallback. Final training applies ROSE to the full selected data
and predicts all original modules on the selected metrics. This is a Python
reimplementation of the approach; selection details, CV, sklearn forests,
and NumPy random sampling may differ from the authors' R implementation.

The demos expose `run_(X, save_path, project_name, model_name, randseed, ...)`
like the other baselines. They bootstrap the dataset, operate on the unlabeled
evaluation sample, compute the existing 13 measures, and append a row to
`save_path/model_name/project_name.csv`. True labels are used only in evaluation.
TCLP's extra options match its algorithm options (except `n_jobs` and
`return_details`). Run from the `code` directory:

```powershell
python .\test\demo_TCL.py
python .\test\demo_TCLP.py
```

Both demos default to all datasets and 100 rounds, with output in `result/TCL`
and `result/TCLP`. Like `demo_MUSDP.py`, edit `save_path` and `Reps` in the
`__main__` configuration block. TCLP also exposes `random_state`,
`n_estimators`, `cv_folds`, and `tune_mtry` there. Results append to existing
CSV files; use a fresh `save_path` for a separate experiment. Unusable rounds
emit warnings and are skipped. TCLP also appends per-round feature names, selection sizes, tuning
scores/status, and generated class counts to `<dataset>_selection.jsonl`.
For a quicker initial TCLP check set `n_estimators = 20` and `cv_folds = 2`.
Dependencies are numpy, pandas, scipy, scikit-learn;
no additional R installation or resampling package is required.

## K-medoids baseline

`algorithms/KMedoids.py` exposes
`KMedoids(metrics, n_clusters=2, random_state=0, standardize=True,
metric='euclidean', max_iter=100, return_details=False)`. Pass numeric metric
columns only; it returns binary pseudo labels (1=predicted defective).
The implementation uses seeded distance-weighted initialization and
best-improving PAM medoid swaps. Representatives are actual input rows.
It needs at least two distinct rows and uses quadratic memory for distances.
PAM finds a local optimum; reaching the iteration limit emits a warning.

Labeling follows Section 3.1 of Yang et al., *On the relative value of
clustering techniques for Unsupervised Effort-Aware Defect Prediction*
(2024): calculate each module's sum of feature values, then average within
each cluster (ASFM). The higher-ASFM cluster is defective. For more than two
clusters, use clusters strictly above the unweighted mean ASFM. Equal-ASFM
clusters are clean. Constant data that cannot form the requested number of
distinct clusters raises ValueError. Optional details include medoid row
positions, cluster membership, ASFM, cost history, and convergence status.

This is a baseline integration rather than an exact experiment replication.
Standardization is enabled explicitly to match this project's metric-based
baselines: both distances and ASFM use standardized metrics. Use
`standardize=False` for raw values. The default distance is
Euclidean; the paper's brief description does not fix all PAM implementation
choices. True labels never enter clustering or ASFM calculation.

`rank_modules(labels, effort)` returns indices sorted by predicted defective
first, then ascending LOC within **both** groups, with stable ties. The demo
uses this order with the shared evaluator and preserves its 13 output measures
and conventions (including its one-based first-defect position for IFA).
It does not search ranking features using true labels.

From the `code` directory:

```powershell
python .\test\demo_KMedoids.py
```

The demo exposes `run_(X, save_path, project_name, model_name, randseed,
standardize=True, max_iter=100)` like the other baselines. It operates on
unlabeled out-of-bootstrap evaluation samples and appends the 13 measures
to `save_path/model_name/project_name.csv`. Like `demo_MUSDP.py`, the script
has an editable configuration block under `if __name__ == '__main__'` and
nested project/round loops. Defaults are all datasets, 100 rounds, and
`result/KMedoids`. Edit `Reps`, `save_path`, `standardize`, and `max_iter`
there to configure a run. Results append to existing CSV files; use a fresh
`save_path` for a separate experiment. Unusable rounds emit a warning.
Dependencies: numpy, pandas, scipy, scikit-learn.

# NOTE 
The software is free for academic use only, and shall not be used, rewritten, or adapted as the basis of a commercial product without first obtaining permission from the authors. 
The authors make no representations about the suitability of this software for any purpose. It is provided "as is" without express or implied warranty.

