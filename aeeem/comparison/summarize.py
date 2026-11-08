"""Summary tables, paired tests, a figure and raw_results.csv from per-round AEEEM results.

raw_results.csv has one row per system, method and bootstrap round, with
every measure. Pd, Pf and AUC_score are empty for merged MUSDP rows, which
the MUSDP runner does not record; n_defective comes from the shared sample.

    python -m aeeem.comparison.summarize result/aeeem_comparison
    python -m aeeem.comparison.summarize result/aeeem_comparison \
        --musdp result/aeeem_musdp/results_raw.csv --output result/aeeem_with_musdp

--musdp merges the MUSDP Part runner's results_raw.csv (test/run_aeeem.py).
Its label1/label2 become MUSDP_v1/MUSDP_v2, one method per view configuration,
paired with the baselines by system and round (both use seed = round).
"""
import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from aeeem.comparison.evaluation import MEASURES

MUSDP_COLUMNS = dict(G='G_mean', AUC='AUC', Popt='Popt', R20='Recall20')
RAW_COLUMNS = ['system', 'method', 'round', 'seed', 'n_test', 'n_defective', 'G_mean', 'AUC',
               'Popt', 'Recall20', 'Pd', 'Pf', 'AUC_score']


def load_musdp(path):
    raw = pd.read_csv(path)
    raw = raw[raw['skipped'] == 0]
    frames = []
    for tag, version in (('label1', 'v1'), ('label2', 'v2')):
        part = raw[['system', 'config', 'rep', 'n_test']].copy()
        part['method'] = f'MUSDP_{version}_' + raw['config']
        for short, measure in MUSDP_COLUMNS.items():
            part[measure] = raw[f'{tag}_{short}']
        part = part.rename(columns={'rep': 'round'}).drop(columns='config')
        frames.append(part.assign(seed=part['round']))  # the MUSDP runner uses seed = round
    return pd.concat(frames, ignore_index=True)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def magnitude(delta):
    d = abs(delta)
    return 'negligible' if d < 0.147 else 'small' if d < 0.33 else 'medium' if d < 0.474 else 'large'


def paired_tests(rounds, reference):
    rows = []
    for (system, method), group in rounds.groupby(['system', 'method']):
        if method == reference:
            continue
        ref = rounds[(rounds.system == system) & (rounds.method == reference)]
        merged = group.merge(ref, on='round', suffixes=('', '_ref'))
        for measure in MEASURES:
            a, b = merged[measure].to_numpy(), merged[measure + '_ref'].to_numpy()
            if len(a) == 0:
                continue
            p = np.nan if np.allclose(a, b) else float(wilcoxon(a, b).pvalue)
            delta = cliffs_delta(a, b)
            rows.append(dict(system=system, method=method, reference=reference, measure=measure,
                             n_pairs=len(a), median_diff=float(np.median(a - b)), p_value=p,
                             cliffs_delta=delta, magnitude=magnitude(delta)))
    return pd.DataFrame(rows)


def summarize(rounds, output, references=('ManualDown',)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    method_order = list(dict.fromkeys(rounds['method']))
    raw = rounds.reindex(columns=RAW_COLUMNS).assign(
        method=lambda d: pd.Categorical(d['method'], method_order, ordered=True))
    # Every method shares the round's evaluation sample, so its defect count is known for MUSDP rows too
    raw['n_defective'] = raw.groupby(['system', 'round'])['n_defective'].transform('max')
    raw.sort_values(['system', 'method', 'round']).to_csv(output / 'raw_results.csv', index=False)
    grouped = rounds.groupby(['system', 'method'])[list(MEASURES)]
    q1, q3 = grouped.quantile(0.25), grouped.quantile(0.75)
    stats = grouped.median().add_suffix('_median').join((q3 - q1).add_suffix('_iqr'))
    stats = stats.join(rounds.groupby(['system', 'method']).size().rename('rounds'))
    stats.to_csv(output / 'median_iqr.csv')

    rank_rows = []
    for measure in MEASURES:
        table = grouped.median()[measure].unstack('method')[method_order]
        table.loc['Median'] = table.median()
        table.to_csv(output / f'{measure}.csv', index_label='System')
        ranks = table.drop(index='Median').rank(axis=1, ascending=False, method='average')
        rank_rows.append(ranks.mean().rename(measure))
    pd.DataFrame(rank_rows).T.assign(mean_rank=lambda d: d.mean(axis=1)).sort_values(
        'mean_rank').to_csv(output / 'mean_ranks.csv', index_label='method')

    tests = pd.concat([paired_tests(rounds, ref) for ref in references
                       if ref in set(rounds['method'])], ignore_index=True)
    tests.to_csv(output / 'paired_tests.csv', index=False)
    plot(rounds, method_order, output / 'boxplots.png')


def plot(rounds, method_order, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    systems = list(dict.fromkeys(rounds['system']))
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    width = 0.8 / len(method_order)
    colors = plt.get_cmap('tab10').colors
    for ax, measure in zip(axes.flat, MEASURES):
        for m, method in enumerate(method_order):
            data = [rounds[(rounds.system == s) & (rounds.method == method)][measure] for s in systems]
            pos = np.arange(len(systems)) - 0.4 + width * (m + 0.5)
            box = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True,
                             showfliers=False, medianprops=dict(color='black', linewidth=1))
            for patch in box['boxes']:
                patch.set_facecolor(colors[m % len(colors)])
                patch.set_alpha(0.75)
        ax.set_title('Recall@20%' if measure == 'Recall20' else measure.replace('_', '-'))
        ax.set_xticks(range(len(systems)), systems)
        ax.grid(axis='y', alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i % len(colors)], alpha=0.75)
               for i in range(len(method_order))]
    fig.legend(handles, method_order, loc='lower center', ncol=min(len(method_order), 6), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('results', help='directory holding rounds.csv')
    parser.add_argument('--musdp', help="MUSDP Part runner's results_raw.csv to merge")
    parser.add_argument('--musdp-configs', nargs='+', default=['full5', 'concat'])
    parser.add_argument('--output', help='defaults to the results directory')
    args = parser.parse_args()
    rounds = pd.read_csv(Path(args.results) / 'rounds.csv')
    references = ['ManualDown']
    if args.musdp:
        musdp = load_musdp(args.musdp)
        keep = musdp['method'].str.replace(r'^MUSDP_v\d_', '', regex=True).isin(args.musdp_configs)
        rounds = pd.concat([musdp[keep], rounds], ignore_index=True)
        references.append('MUSDP_v1_full5')
    summarize(rounds, args.output or args.results, references)


if __name__ == '__main__':
    main()
