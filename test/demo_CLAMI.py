"""Run CLAMI using the project's bootstrap evaluation and 13 measures."""
import os
import sys
import warnings
from pathlib import Path

# Direct script execution adds test/, rather than code/, to the import path.
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from algorithms.CLA import CLAMI
import numpy as np
import pandas as pd
from utilities import performanceMeasure, rankMeasure
from utilities.bootstrapCV import outofsample_bootstrap
from utilities.File import create_dir, save_results


def run_(X, save_path, project_name, model_name, randseed, percentile=50):
    """Bootstrap, evaluate, and append one round's 13 measures to CSV.

    X contains metric columns and a final defect-label column. True labels
    are used only for evaluation. save_path is the parent results directory.
    """
    print(f'{project_name}: -> {model_name} {randseed + 1} round Start!')
    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X, randseed)
    if np.unique(test_label).size < 2:
        raise ValueError('Evaluation sample lacks both classes')
    test_LOC = test_data['CountLineCode'].to_numpy()
    predict_y = CLAMI(test_data, percentile=percentile)

    # calculate non-effort-aware classification measures
    AUC, G_mean, precision, recall, pf, F1, MCC = performanceMeasure.get_measure(
        test_label, predict_y)
    # calculate cost-effectiveness measures
    Popt, recall_effort, precision_effort, F1_effort, PMI, IFA = rankMeasure.rank_measure(
        predict_y, test_LOC, test_label)
    measure = [AUC, G_mean, precision, recall, pf, F1, MCC,
               Popt, recall_effort, precision_effort, F1_effort, PMI, IFA]
    fres = create_dir(str(Path(save_path) / model_name))
    save_results(str(Path(fres) / project_name), measure)
    return measure


if __name__ == '__main__':
    save_path = str(CODE_DIR / 'result') + os.sep
    model_name = 'CLAMI'
    Reps = 100
    percentile = 50

    path = CODE_DIR / 'data'
    project_names = sorted(p.name for p in path.glob('*.csv'))
    pro_num = len(project_names)

    for i in range(pro_num):
        project_name = project_names[i]
        file = os.path.join(path, project_name)
        data = pd.read_csv(file)
        project_name = project_name[:-4]

        for loop in range(Reps):
            try:
                run_(data, save_path, project_name, model_name, loop, percentile=percentile)
            except ValueError as error:
                warnings.warn(project_name + ': round ' + str(loop + 1) + ' skipped: ' + str(error))

    print('done!')
