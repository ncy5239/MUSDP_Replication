"""Load an AEEEM system as a feature matrix, defect labels, LOC and semantic views.

The five views follow the metric families of D'Ambros et al. (2010) and match
the prefixes used by the MUSDP Part runner (test/run_aeeem.py):

    source       ck_oo_*            17 CK and object-oriented source metrics
    churn        WCHU_*             17 weighted churn of the source metrics
    entropy_src  LDHH_*             17 linearly decayed entropy of the source metrics
    entropy_chg  Cvs*                5 entropy of changes
    prev_defects *BugsFoundUntil     5 previous-defect counts
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'AEEEM'
SYSTEMS = ('EQ', 'JDT', 'LC', 'ML', 'PDE')
LOC_COLUMN = 'ck_oo_numberOfLinesOfCode'
CHURN_COLUMN = 'WCHU_numberOfLinesOfCode'
LABEL_COLUMN = 'class'
N_FEATURES = 61

VIEWS = {
    'source': lambda c: c.startswith('ck_oo_'),
    'churn': lambda c: c.startswith('WCHU_'),
    'entropy_src': lambda c: c.startswith('LDHH_'),
    'entropy_chg': lambda c: c.startswith('Cvs'),
    'prev_defects': lambda c: 'BugsFoundUntil' in c,
}


@dataclass(frozen=True)
class AEEEMSystem:
    name: str
    features: pd.DataFrame
    label: np.ndarray
    loc: np.ndarray

    def view(self, name):
        return self.features[[c for c in self.features.columns if VIEWS[name](c)]]

    def frame(self):
        """Features with the label as the last column, the layout outofsample_bootstrap expects."""
        return self.features.assign(bug=self.label)


def load_system(name, data_dir=DATA_DIR):
    raw, _ = arff.loadarff(Path(data_dir) / f'{name}.arff')
    df = pd.DataFrame(raw)
    df.columns = [c.rstrip(':') for c in df.columns]
    classes = df.pop(LABEL_COLUMN).str.decode('utf-8')
    if set(classes) != {'buggy', 'clean'}:
        raise ValueError(f'{name}: unexpected class values {sorted(set(classes))}')
    features = df.astype(float).reset_index(drop=True)
    if features.shape[1] != N_FEATURES:
        raise ValueError(f'{name}: expected {N_FEATURES} metrics, found {features.shape[1]}')
    owners = [[v for v, match in VIEWS.items() if match(c)] for c in features.columns]
    unassigned = [c for c, o in zip(features.columns, owners) if len(o) != 1]
    if unassigned:
        raise ValueError(f'{name}: metrics not in exactly one view: {unassigned}')
    if not np.isfinite(features.to_numpy()).all() or (features[LOC_COLUMN] < 0).any():
        raise ValueError(f'{name}: metrics must be finite and LOC nonnegative')
    return AEEEMSystem(name=name, features=features,
                       label=(classes == 'buggy').to_numpy(dtype=int),
                       loc=features[LOC_COLUMN].to_numpy(dtype=float))
