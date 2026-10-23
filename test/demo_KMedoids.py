"""Bootstrap k-medoids with the project's existing 13 evaluation measures."""
import os
import sys
import warnings
from pathlib import Path

# Direct script execution adds test/, rather than code/, to the import path.
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
from algorithms.KMedoids import KMedoids, rank_modules
from utilities import performanceMeasure, rankMeasure
from utilities.bootstrapCV import outofsample_bootstrap
from utilities.File import create_dir, save_results


def run_(X, save_path, project_name, model_name, randseed, standardize=True,
         max_iter=100):
    """Cluster unlabeled out-of-bootstrap metrics; true labels only evaluate."""
    print(project_name + ': -> ' + model_name + ' ' + str(randseed + 1) + ' round Start!')

    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X, randseed)
    if np.unique(test_label).size < 2:
        raise ValueError('Evaluation sample lacks both classes')
    test_LOC = test_data['CountLineCode'].to_numpy()

    # K-medoids clustering and ASFM cluster labeling
    clus_label = KMedoids(test_data, random_state=randseed,
                         standardize=standardize, max_iter=max_iter)
    predict_y = clus_label

    # calculate non-effort-aware classification measures
    AUC, G_mean, precision, recall, pf, F1, MCC = performanceMeasure.get_measure(test_label, predict_y)

    # calculate cost-effectiveness measures
    order = rank_modules(predict_y, test_LOC)
    # The shared evaluator uses stable sorting; preordering also sorts the
    # clean group, whose zero predicted densities would otherwise all tie.
    Popt, recall_effort, precision_effort, F1_effort, PMI, IFA = rankMeasure.rank_measure(
        predict_y[order], test_LOC[order], test_label[order])

    measure = [AUC, G_mean, precision, recall, pf, F1, MCC, Popt, recall_effort, precision_effort, F1_effort, PMI, IFA]

    fres = create_dir(str(Path(save_path) / model_name))
    save_results(str(Path(fres) / project_name), measure)
    return measure


if __name__ == '__main__':
    save_path = str(CODE_DIR / 'result') + os.sep
    model_name = 'KMedoids'
    Reps = 100
    standardize = True
    max_iter = 100

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
                run_(data, save_path, project_name, model_name, loop,
                     standardize=standardize, max_iter=max_iter)
            except ValueError as error:
                warnings.warn(project_name + ': round ' + str(loop + 1) + ' skipped: ' + str(error))

    print('done!')
