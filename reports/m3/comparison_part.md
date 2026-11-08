# M3: Comparison Part (Emmanuel Omole, 101004432)

Text for the group report, grouped under the M3 headings. The LaTeX version
is `comparison_part.tex`; the figure is `fig_aeeem_comparison.png`.

## Methods executed (Comparison Part)

**Data.** AEEEM: EQ (324 classes, 39.8% defective), JDT (997, 20.7%), LC
(691, 9.3%), ML (1862, 13.2%) and PDE (1497, 14.0%). Each class has 61
metrics in five families: 17 CK and object-oriented source metrics, 17
weighted churn metrics (WCHU), 17 entropy-of-source metrics (LDHH), 5
entropy-of-change metrics and 5 previous-defect counts. LOC is
`ck_oo_numberOfLinesOfCode`, the AEEEM counterpart of `CountLineCode` in the
original study. It is the ranking metric of ManualDown and ManualUp and the
effort for Popt and Recall@20%.

**Baselines.**

- SC: the original spectral clustering on all 61 metrics, z-scored, k = 2.
  The cluster with the larger mean row sum is defective. No AutoSpearman, as
  in the original baseline.
- ManualDown / ManualUp: rank by LOC, descending / ascending. The top half is
  defective.
- Churn (minimal churn baseline): rank by `WCHU_numberOfLinesOfCode`, the
  weighted churn of LOC; the top half is defective. It uses ManualDown's
  cut-off and breaks ties by larger LOC, so it differs from ManualDown only in
  the ranking metric. A fixed tie rule is needed because 27% (PDE) to 54% (LC)
  of classes have zero churn.
- ChurnSet: the same rule, ranking by the sum of the 17 z-scored WCHU metrics.

MUSDP results in this section come from Member 2's runner
(`test/run_aeeem.py`), with its MUSDP logic unchanged, in three configurations: five views
(MUSDP-5V), all metrics concatenated into one view (MUSDP-concat) and the
churn view alone (MUSDP-churn).

**Protocol.** 100 out-of-sample bootstrap rounds per system, as in the
original study. Every method uses seed = round, so in each round all methods
are scored on the same held-out sample and results pair across methods.
Methods see only the unlabeled held-out sample; labels are used for scoring
only. We report the median over rounds, the IQR, the mean rank over the five
systems, and paired Wilcoxon signed-rank tests with Cliff's delta.

**Evaluation.** One function scores every method with the original study's
measure code. G-mean and AUC use the binary label; for a binary label,
AUC = (Pd + 1 - Pf) / 2, which is how the paper computes AUC. Popt and
Recall@20% use LOC as effort. Ranking methods are inspected in score order.
Binary methods (SC, MUSDP) are inspected defective first, then by predicted
defect density, the original convention. Recall@20% counts the class at which
cumulative LOC reaches 20%. For ranking methods we also record the rank AUC
of the score.

**Tools.** Python 3.11, numpy 1.26, scikit-learn, scipy, pytest. Code is in
`aeeem/` in the repository.

## Validation of the measures

- Hand-worked cases for every measure, for example a four-class ranking with
  Popt = 6/7 and Recall@20% = 0.5 computed on paper.
- Popt and Recall@20% re-implemented from their definitions agree with the
  original code on 200 random cases.
- End-to-end: SC, ManualDown and ManualUp run through our pipeline on the
  original 28 releases (100 rounds) reproduce all 168 per-release median Pd
  and Pf values in Tables 5 and 6 of the online supplementary to within
  0.0005, the rounding of the published three-decimal values. G-mean and
  AUC are functions of Pd and Pf in each round, so this checks the binary
  side of the pipeline end to end.
- Issue found: `rank_measure` adds 1 to LOC when `0 in effort`. For a pandas
  Series this tests the index, not the values, so `demo_SC.py` corrects zero
  LOC only when index 0 happens to be in the test sample. Our pipeline passes
  arrays. This can affect Member 1's reproduced Popt and Recall@20% for SC.

## Preliminary findings (new context: AEEEM)

Table I. Median over the five systems of the per-system medians (mean rank
over systems in brackets, 1 = best).

| Method | G-mean | AUC | Popt | Recall@20% |
|---|---|---|---|---|
| MUSDP-5V | **0.711** (2.6) | **0.716** (2.6) | 0.600 (3.2) | 0.386 (2.8) |
| MUSDP-concat | 0.695 (2.8) | 0.697 (3.0) | 0.595 (3.0) | 0.394 (2.8) |
| MUSDP-churn | 0.655 (5.6) | 0.656 (5.6) | 0.593 (3.8) | **0.409** (2.8) |
| SC | 0.695 (2.4) | 0.705 (2.4) | 0.573 (4.4) | 0.358 (3.6) |
| ManualDown | 0.657 (5.2) | 0.667 (5.2) | 0.278 (8.0) | 0.059 (8.0) |
| ManualUp | 0.312 (8.0) | 0.334 (8.0) | **0.722** (1.0) | 0.385 (3.0) |
| Churn | 0.662 (3.6) | 0.677 (3.4) | 0.425 (6.8) | 0.162 (6.6) |
| ChurnSet | 0.642 (5.8) | 0.655 (5.8) | 0.421 (5.8) | 0.151 (6.4) |

Table II. Median G-mean and Popt per system.

| Method | EQ | JDT | LC | ML | PDE |
|---|---|---|---|---|---|
| G-mean MUSDP-5V | 0.711 | 0.744 | 0.734 | 0.600 | 0.684 |
| G-mean MUSDP-concat | 0.724 | 0.710 | 0.695 | 0.615 | 0.668 |
| G-mean SC | 0.695 | 0.759 | 0.735 | 0.621 | 0.678 |
| G-mean ManualDown | 0.687 | 0.671 | 0.594 | 0.626 | 0.657 |
| G-mean Churn | 0.731 | 0.700 | 0.662 | 0.614 | 0.623 |
| Popt MUSDP-5V | 0.581 | 0.621 | 0.736 | 0.511 | 0.600 |
| Popt SC | 0.519 | 0.605 | 0.722 | 0.496 | 0.573 |
| Popt ManualUp | 0.835 | 0.722 | 0.766 | 0.689 | 0.632 |
| Popt Churn | 0.276 | 0.375 | 0.652 | 0.425 | 0.468 |

Fig. 1 (`fig_aeeem_comparison.png`): distribution over 100 rounds per system.

1. MUSDP-5V has the best overall mean rank and the highest median G-mean
   (0.711) and AUC (0.716), but the lead is small and not uniform. SC is
   0.016 G-mean behind overall, beats MUSDP-5V on JDT and ML (Cliff's delta
   0.28 and 0.42, p < 0.05) and does not differ on LC.
2. Multiview structure: MUSDP-concat, which keeps the same metrics in one
   view, loses to MUSDP-5V on JDT, LC and PDE (G-mean 0.018 to 0.037 lower,
   delta -0.37 to -0.69), is level on EQ (negligible delta) and is better on
   ML. The five-view structure helps on three of the five systems.
3. A single churn metric is close on classification. Churn reaches a median
   G-mean of 0.662, 93% of MUSDP-5V. It beats ManualDown on EQ, JDT and LC
   (delta 0.67 to 0.90) and MUSDP-5V on EQ (0.731 vs 0.711) and ML (0.614 vs
   0.600). Its rank AUC is 0.68 to 0.81. Summing all 17 churn metrics
   (ChurnSet) does not improve on the single metric.
4. The measures disagree. ManualUp has the best Popt on every system (median
   0.722) and the worst G-mean (0.312). Churn and ManualDown inspect large
   classes first and score low on Recall@20% (0.162 and 0.059). The same churn
   metrics clustered by MUSDP (MUSDP-churn) reach Popt 0.593.
5. Across systems, ML has the lowest best G-mean (0.626, by ManualDown), and
   no method leads on all five systems. MUSDP's two labelling strategies
   produced identical labels in all 1500 rounds on AEEEM: majority voting
   always chose the smaller cluster, so MUSDP_v1 and MUSDP_v2 are one row here.

## Interpretation and validity

The evidence supports a weaker version of the paper's claim on AEEEM: MUSDP
has the best median classification scores, but its overall margin is 0.016
G-mean over SC and 0.049 over a one-metric churn ranking, and both baselines
beat it on some systems.

The strongest threat is class size. LOC is the effort in Popt and
Recall@20%, and churn and most metrics grow with size. Effort-aware scores
mostly reward inspecting small classes first: ManualUp wins Popt by doing only
that, and binary methods get part of the same benefit from the density
ordering. Effort-aware differences between MUSDP and the baselines may
therefore come from size ordering rather than from the clustering.

Other threats: the fixed 50% cut-off sets the G-mean and AUC of the ranking
methods; held-out samples overlap between bootstrap rounds, so Wilcoxon
p-values over rounds are optimistic and the effect sizes are the more
reliable figure; the MUSDP results are preliminary runs of Member 2's code.

## Remaining work (to M4)

- Merge Member 2's final MUSDP configurations (every single view and
  drop-one views) through `aeeem.comparison.summarize --musdp`.
- Rank methods with NPSKESD, the paper's test, alongside Wilcoxon.
- Control for size: a churn-density baseline (churn / LOC) and results by size
  stratum; test the sensitivity to the 50% cut-off.

## Contributions (Emmanuel Omole)

- `aeeem/loader.py`: AEEEM loader from the ARFF files with a check that the
  five views cover all 61 metrics.
- `aeeem/comparison/`: SC, ManualDown, ManualUp, Churn and ChurnSet on AEEEM;
  one evaluation for all methods; runner; summary tables, paired tests and
  figure.
- Validation: 212 tests and the end-to-end check against the paper (168 of
  168 published values reproduced).
- Ran the 100-round experiments for all baselines and the three MUSDP
  comparison configurations. Found the `rank_measure` Series issue. Made
  `test/run_aeeem.py` read the ARFF files through the loader and run on
  macOS (it read CSV files the repository did not contain and needed the
  fork start method), with identical results.
