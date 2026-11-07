"""Check the four measures and the baselines on hand-built cases before the real runs.

    python -m pytest aeeem/comparison/validation -q

The hand cases are worked out on paper in the comments. The reference
implementations at the bottom are written from the measure definitions, not
from utilities/rankMeasure.py, and compared with it on random samples.
"""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[3]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
import pytest

from aeeem.comparison import baselines
from aeeem.comparison.baselines import Prediction
from aeeem.comparison.evaluation import evaluate
from aeeem.loader import N_FEATURES, SYSTEMS, VIEWS, load_system
from utilities import rankMeasure

# Four modules: A(LOC 10, buggy), B(20, clean), C(30, buggy), D(40, clean); 100 LOC in total.
LOC = np.array([10.0, 20.0, 30.0, 40.0])
TRUTH = np.array([1, 0, 1, 0])


def test_gmean_and_auc_from_confusion_matrix():
    # tp=2, fn=1, fp=1, tn=2 -> Pd = 2/3, Pf = 1/3
    # G-mean = sqrt(2/3 * 2/3) = 2/3; AUC on a binary label = (Pd + 1 - Pf) / 2 = 2/3
    truth = np.array([1, 1, 1, 0, 0, 0])
    label = np.array([1, 1, 0, 1, 0, 0])
    v = evaluate(truth, Prediction(label=label), np.ones(6))
    assert v['Pd'] == pytest.approx(2 / 3)
    assert v['Pf'] == pytest.approx(1 / 3)
    assert v['G_mean'] == pytest.approx(2 / 3)
    assert v['AUC'] == pytest.approx(2 / 3)


def test_popt_and_recall20_for_a_ranking():
    # ManualUp order A, B, C, D: cumulative LOC .1 .3 .6 1.0, cumulative bugs .5 .5 1 1.
    # Model area (trapezoids from the origin) = .025 + .1 + .225 + .4 = .75
    # Optimal order A, C, B, D (density desc, then LOC asc): .025 + .225 + .2 + .4 = .85
    # Worst order D, B, C, A (density asc, then LOC desc):   0 + 0 + .075 + .075 = .15
    # Popt = 1 - (.85 - .75) / (.85 - .15) = 6/7
    # Recall@20%: the first module reaching 20% of LOC is B (30%), bugs found 1 of 2 -> .5
    v = evaluate(TRUTH, baselines.manual_up(LOC), LOC)
    assert v['Popt'] == pytest.approx(6 / 7)
    assert v['Recall20'] == pytest.approx(0.5)
    assert v['AUC_score'] == pytest.approx(0.75)  # pairs (A>B, A>D, C>D) ranked right, (B>C) wrong


def test_binary_prediction_is_ranked_defective_first_then_by_density():
    # Predicted defective A and C, inspected A then C (1/10 > 1/30), then B, D.
    # That is the optimal order, so Popt = 1; 40% of LOC finds both bugs -> Recall@20% = 1.
    v = evaluate(TRUTH, Prediction(label=np.array([1, 0, 1, 0])), LOC)
    assert v['Popt'] == pytest.approx(1.0)
    assert v['Recall20'] == pytest.approx(1.0)
    assert v['G_mean'] == pytest.approx(1.0)


def test_manual_down_is_the_reverse_ranking():
    # Order D, C, B, A: cumulative LOC .4 .7 .9 1.0, bugs 0 .5 .5 1.
    # Area = 0 + .075 + .1 + .075 = .25; Popt = 1 - (.85 - .25) / .7 = 1/7
    v = evaluate(TRUTH, baselines.manual_down(LOC), LOC)
    assert v['Popt'] == pytest.approx(1 / 7)
    assert v['Recall20'] == pytest.approx(0.0)
    assert list(baselines.manual_down(LOC).label) == [0, 0, 1, 1]


def test_zero_loc_adds_one_to_every_module():
    loc = np.array([0.0, 20.0, 30.0, 40.0])
    pred = Prediction(label=np.array([1, 0, 1, 0]))
    assert evaluate(TRUTH, pred, loc) == evaluate(TRUTH, pred, loc + 1)


def test_rank_measure_series_pitfall_is_real():
    """rank_measure's `0 in effort` looks at a Series' index, not its values.

    With an index that has no label 0, a zero LOC is left as 0 and the
    measures change. evaluate() always passes arrays, so it is not affected.
    """
    loc = pd.Series([0.0, 20.0, 30.0, 40.0], index=[5, 6, 7, 8])
    score = np.array([4.0, 3.0, 2.0, 1.0])
    as_series = rankMeasure.rank_measure(score, loc, TRUTH, opt=1)
    as_array = rankMeasure.rank_measure(score, loc.to_numpy(), TRUTH, opt=1)
    assert as_series[0] != pytest.approx(as_array[0])


def test_churn_ranks_by_churn_then_larger_loc():
    churn = np.array([0.0, 5.0, 0.0, 5.0, 1.0])
    loc = np.array([10.0, 1.0, 50.0, 3.0, 2.0])
    pred = baselines.churn(churn, loc)
    order = np.argsort(-pred.score)
    assert list(order) == [3, 1, 4, 2, 0]
    assert list(pred.label) == [0, 1, 0, 1, 0]  # floor(5/2) = 2 defective


@pytest.mark.parametrize('name', SYSTEMS)
def test_loader_shapes_and_views(name):
    system = load_system(name)
    assert system.features.shape[1] == N_FEATURES
    sizes = {v: system.view(v).shape[1] for v in VIEWS}
    assert sizes == dict(source=17, churn=17, entropy_src=17, entropy_chg=5, prev_defects=5)
    assert set(np.unique(system.label)) == {0, 1}
    assert len(system.loc) == len(system.label) == len(system.features)


# Reference implementations written from the definitions.

def reference_curve(order, effort, truth):
    x = np.concatenate([[0], np.cumsum(effort[order]) / effort.sum()])
    y = np.concatenate([[0], np.cumsum(truth[order]) / truth.sum()])
    return x, y


def reference_popt(order, effort, truth):
    n = len(truth)
    density = truth / effort
    optimal = sorted(range(n), key=lambda i: (-density[i], effort[i]))
    worst = sorted(range(n), key=lambda i: (density[i], -effort[i]))
    area = lambda o: np.trapz(*reference_curve(np.array(o), effort, truth)[::-1])
    return 1 - (area(optimal) - area(order)) / (area(optimal) - area(worst))


def reference_recall20(order, effort, truth):
    x, y = reference_curve(order, effort, truth)
    return y[np.argmax(x[1:] >= 0.2) + 1]


@pytest.mark.parametrize('seed', range(200))
def test_rank_measure_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 60))
    effort = rng.integers(1, 500, n).astype(float)
    truth = rng.integers(0, 2, n)
    if truth.sum() == 0:
        truth[0] = 1
    score = rng.permutation(n).astype(float)  # distinct scores -> one order
    order = np.argsort(-score, kind='stable')
    popt, recall20, *_ = rankMeasure.rank_measure(score, effort, truth, opt=1)
    assert popt == pytest.approx(reference_popt(order, effort, truth))
    assert recall20 == pytest.approx(reference_recall20(order, effort, truth))
