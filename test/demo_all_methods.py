"""Run all defect-prediction methods and export four dataset-by-method tables."""
import json
import sys
import warnings
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
from sklearn import preprocessing
from algorithms.CLA import CLA, CLAMI
from algorithms.KMedoids import KMedoids, rank_modules
from algorithms.ManualDownUp import manualDown, manualUp
from algorithms.SC import SC
from algorithms.TCL import TCL, TCLP
from algorithms.coreg import MultiviewCoRegSpectralClustering
from algorithms.coreg.mean_replacement_importance import predict_labels
from utilities import performanceMeasure, rankMeasure
from utilities.AutoSpearman import AutoSpearman
from utilities.bootstrapCV import outofsample_bootstrap

METHODS = ('ManualUp', 'ManualDown', 'SC', 'CLA', 'CLAMI', 'TCL', 'TCLP',
           'KMedoids', 'MUSDP_label1', 'MUSDP_label2')
MEASURES = ('G_mean', 'AUC', 'Popt', 'Recall20')


def _evaluate(truth, prediction, effort, method):
    """Use existing metric conventions, including the 20%-LOC crossing module."""
    auc, g_mean, *_ = performanceMeasure.get_measure(truth, prediction)
    if method in ('ManualUp', 'ManualDown'):
        adjusted = effort + 1 if np.any(effort == 0) else effort
        score = 1 / adjusted if method == 'ManualUp' else effort
        popt, recall20, *_ = rankMeasure.rank_measure(score, adjusted, truth, opt=1)
    else:
        order = (rank_modules(prediction, effort) if method == 'KMedoids'
                 else np.arange(len(truth)))
        popt, recall20, *_ = rankMeasure.rank_measure(
            prediction[order], effort[order], truth[order])
    values = dict(zip(MEASURES, map(float, (g_mean, auc, popt, recall20))))
    if not np.isfinite(list(values.values())).all():
        raise ValueError('Evaluation produced a nonfinite measure')
    return values


def run_all_methods(data_dir=None, save_path=None, Reps=100, datasets=None,
                    methods=None, base_seed=0, percentile=50,
                    n_estimators=500, cv_folds=10, tune_mtry=True,
                    standardize=True, max_iter=100, v_lambda=1):
    """Return four DataFrames and write G_mean/AUC/Popt/Recall20.csv.

    Rows are dataset stems, columns are methods, and cells are arithmetic means
    across successful rounds (no rounding before export). All methods share
    the same out-of-bootstrap sample for each dataset/round. Actual labels
    are used only for evaluation. Failed method-rounds are excluded, never
    replaced with zeros; cells with no successful rounds are NaN.

    A fresh save_path is required. JSON settings, round values, successful
    counts, and failure logs accompany the four CSV files. Default runs all
    datasets and methods. MUSDP expects the repository's 65-metric schema.
    """
    if not isinstance(Reps, int) or Reps < 1:
        raise ValueError('Reps must be a positive integer')
    if not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError('base_seed must be a nonnegative integer')
    if not 0 <= percentile <= 100 or n_estimators < 1 or cv_folds < 2 or max_iter < 1:
        raise ValueError('Invalid percentile, n_estimators, cv_folds, or max_iter')
    data_dir = Path(data_dir) if data_dir is not None else CODE_DIR / 'data'
    save_path = Path(save_path) if save_path is not None else CODE_DIR / 'result' / 'all_methods'
    methods = list(METHODS if methods is None else methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError('methods must be unique names from METHODS')
    paths = sorted(data_dir.glob('*.csv'))
    if datasets is not None:
        requested = list(datasets)
        if not requested or set(requested) - {p.stem for p in paths}:
            raise ValueError('datasets must name existing CSV stems')
        paths = [p for p in paths if p.stem in requested]
    if not paths:
        raise ValueError('No dataset CSV files found')
    save_path.mkdir(parents=True, exist_ok=False)
    settings = dict(data_dir=str(data_dir.resolve()), datasets=[p.stem for p in paths],
                    methods=methods, Reps=Reps, base_seed=base_seed,
                    percentile=percentile, n_estimators=n_estimators,
                    cv_folds=cv_folds, tune_mtry=tune_mtry, standardize=standardize,
                    max_iter=max_iter, v_lambda=v_lambda, aggregation='arithmetic mean',
                    bootstrap_seed='base_seed + zero-based round',
                    AUC='binary predicted labels, matching existing demos',
                    Recall20='includes first module reaching or exceeding 20% LOC',
                    MUSDP_label1='majority vote across independently labeled views')
    (save_path / 'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    tables = {name: pd.DataFrame(np.nan, index=[p.stem for p in paths], columns=methods)
              for name in MEASURES}
    counts = {p.stem: {method: 0 for method in methods} for p in paths}

    with (save_path / 'rounds.jsonl').open('w', encoding='utf-8') as rounds, \
            (save_path / 'failures.jsonl').open('w', encoding='utf-8') as failures:
        def failed(dataset, round_number, method, error):
            failures.write(json.dumps(dict(dataset=dataset, round=round_number,
                                            method=method, reason=str(error))) + '\n')
            failures.flush()
            warnings.warn(f'{dataset}: {method} round {round_number + 1} skipped: {error}')

        for path in paths:
            data = pd.read_csv(path).reset_index(drop=True)
            if 'CountLineCode' not in data.columns[:-1]:
                raise ValueError(f'{path.name}: CountLineCode must be a metric column')
            metrics = data.iloc[:, :-1].to_numpy(dtype=float)
            truth_all = data.iloc[:, -1].to_numpy(dtype=float)
            if (not np.isfinite(metrics).all() or not np.isfinite(truth_all).all()
                    or (truth_all < 0).any() or (data['CountLineCode'] < 0).any()):
                raise ValueError(f'{path.name}: metrics/labels must be finite; labels and LOC nonnegative')
            views, view_error = None, None
            if any(m.startswith('MUSDP_') for m in methods):
                try:
                    if metrics.shape[1] != 65:
                        raise ValueError('MUSDP requires the repository 65-metric view layout')
                    # Match the demos: unsupervised feature selection before bootstrap.
                    views = [AutoSpearman(data.iloc[:, start:end])
                             for start, end in ((0, 54), (54, 59), (59, 65))]
                    if any(view.shape[1] == 0 for view in views):
                        raise ValueError('MUSDP feature selection returned an empty view')
                except (ValueError, np.linalg.LinAlgError) as error:
                    view_error = error
            records = []
            for loop in range(Reps):
                seed = base_seed + loop
                _, _, test_data, truth, _, indices = outofsample_bootstrap(data.copy(), seed)
                effort = test_data['CountLineCode'].to_numpy(dtype=float)
                if len(truth) < 2 or np.unique(truth).size < 2:
                    for method in methods:
                        failed(path.stem, loop, method, 'Evaluation sample lacks both classes')
                    continue
                musdp_predictions, musdp_error = None, view_error
                for method in methods:
                    print(f'{path.stem}: -> {method} {loop + 1} round Start!', flush=True)
                    try:
                        if method.startswith('MUSDP_'):
                            if musdp_error is not None:
                                raise ValueError(str(musdp_error))
                            if musdp_predictions is None:
                                sampled = [preprocessing.scale(view.iloc[indices]) for view in views]
                                model = MultiviewCoRegSpectralClustering(
                                    n_clusters=2, v_lambda=v_lambda, max_iter=10, random_state=seed)
                                musdp_predictions = predict_labels(sampled, model.fit_predict(sampled))
                            prediction = musdp_predictions[int(method[-1]) - 1]
                        elif method == 'ManualUp':
                            prediction = manualUp(effort.copy())
                        elif method == 'ManualDown':
                            prediction = manualDown(effort.copy())
                        elif method == 'SC':
                            prediction = SC(test_data)
                        elif method == 'CLA':
                            prediction = CLA(test_data, percentile=percentile)
                        elif method == 'CLAMI':
                            prediction = CLAMI(test_data, percentile=percentile, random_state=seed)
                        elif method == 'TCL':
                            prediction = TCL(test_data)
                        elif method == 'TCLP':
                            prediction = TCLP(test_data, random_state=seed,
                                              n_estimators=n_estimators, cv_folds=cv_folds,
                                              tune_mtry=tune_mtry)
                        else:
                            prediction = KMedoids(test_data, random_state=seed,
                                                 standardize=standardize, max_iter=max_iter)
                        values = _evaluate(truth, np.asarray(prediction), effort, method)
                    except (ValueError, np.linalg.LinAlgError) as error:
                        if method.startswith('MUSDP_'):
                            musdp_error = error
                        failed(path.stem, loop, method, error)
                        continue
                    counts[path.stem][method] += 1
                    records.append(dict(method=method, **values))
                    rounds.write(json.dumps(dict(dataset=path.stem, round=loop,
                                                  seed=seed, method=method, **values)) + '\n')
                    rounds.flush()
                # Checkpoint summaries so a long run retains completed rounds.
                if records:
                    means = pd.DataFrame(records).groupby('method')[list(MEASURES)].mean()
                    for measure, table in tables.items():
                        for method in means.index:
                            table.loc[path.stem, method] = means.loc[method, measure]
                for measure, table in tables.items():
                    table.to_csv(save_path / (measure + '.csv'), index_label='Dataset')
                (save_path / 'successful_rounds.json').write_text(
                    json.dumps(counts, indent=2), encoding='utf-8')
    # Also create valid empty summaries when all bootstrap samples were unusable.
    for measure, table in tables.items():
        table.to_csv(save_path / (measure + '.csv'), index_label='Dataset')
    (save_path / 'successful_rounds.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(f'Results: {save_path.resolve()}')
    return tables


if __name__ == '__main__':
    save_path = CODE_DIR / 'result' / 'all_methods'
    Reps = 100
    datasets = None  # Example: ['activemq-5.0.0']; None runs all datasets.
    methods = None  # None runs every method in METHODS.
    run_all_methods(save_path=save_path, Reps=Reps, datasets=datasets, methods=methods)
