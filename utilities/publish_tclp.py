"""Publish genuine TCLP batch results into the existing combined result files."""
import csv
import json
import math
import shutil
import statistics
from collections import defaultdict
from datetime import datetime, timezone

MEASURES = ('G_mean', 'AUC', 'Popt', 'Recall20')


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()] if path.exists() else []


def write(path, text):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(text,encoding='utf-8');temporary.replace(path)


def publish(result, batch):
    settings=json.loads((result/'settings.json').read_text())
    datasets,methods=settings['datasets'],settings['methods']
    key=lambda r:(r['dataset'],r['method'],int(r['seed']))
    canonical=read_rows(result/'rounds.jsonl');indexed={key(r):r for r in canonical}
    if len(indexed)!=len(canonical):raise ValueError('Duplicate seeds in canonical results')
    batch_settings=json.loads((batch/'settings.json').read_text())
    added=[]
    for original in read_rows(batch/'rounds.jsonl'):
        row=dict(original);row['round']=row['seed'];identity=key(row)
        if row['method']!='TCLP' or row['dataset'] not in datasets or not 0<=row['seed']<100:raise ValueError('Invalid TCLP identity')
        if not all(math.isfinite(row[m]) and 0<=row[m]<=1 for m in MEASURES):raise ValueError('Invalid TCLP measure')
        if identity in indexed:
            if any(not math.isclose(indexed[identity][m],row[m],rel_tol=1e-10,abs_tol=1e-12) for m in MEASURES):raise ValueError('Conflicting TCLP seed '+str(identity))
            continue
        indexed[identity]=row;added.append(row)
    if added:
        canonical=sorted(indexed.values(),key=lambda r:(datasets.index(r['dataset']),r['seed'],methods.index(r['method'])))
        write(result/'rounds.jsonl',''.join(json.dumps(r)+'\n' for r in canonical))
        with (result/'provenance.jsonl').open('a',encoding='utf-8') as f:
            for r in added:f.write(json.dumps(dict(dataset=r['dataset'],method='TCLP',seed=r['seed'],selected_source=str(batch),original_round=r['seed']-batch_settings['base_seed'],present_in_both=False))+'\n')
    counts={d:{m:0 for m in methods} for d in datasets};values=defaultdict(list)
    for row in canonical:
        counts[row['dataset']][row['method']]+=1
        for measure in MEASURES:values[row['dataset'],row['method'],measure].append(row[measure])
    for measure in MEASURES:
        for partial in (False,True):
            path=result/(measure+('_partial.csv' if partial else '.csv'));temporary=path.with_name(path.name+'.tmp')
            with temporary.open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f);w.writerow(['Dataset',*methods])
                for d in datasets:w.writerow([d,*[statistics.median(values[d,m,measure]) if counts[d][m]>0 and (partial or counts[d][m]==100) else '' for m in methods]])
            temporary.replace(path)
    write(result/'successful_rounds.json',json.dumps(counts,indent=2))
    path=result/'Repetitions.csv';temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['Dataset',*methods]);w.writerows([d,*[counts[d][m] for m in methods]] for d in datasets)
    temporary.replace(path)
    failures=read_rows(result/'failures.jsonl');seen={(r['dataset'],r['method'],r.get('seed'),r.get('reason')) for r in failures}
    for original in read_rows(batch/'failures.jsonl'):
        row=dict(original);row['seed']=batch_settings['base_seed']+row['round'];row['round']=row['seed']
        identity=(row['dataset'],row['method'],row['seed'],row.get('reason'))
        if identity not in seen:failures.append(row);seen.add(identity)
    write(result/'failures.jsonl',''.join(json.dumps(r)+'\n' for r in failures))
    names=['rounds.jsonl','provenance.jsonl','failures.jsonl','successful_rounds.json','Repetitions.csv',*[m+s+'.csv' for m in MEASURES for s in ('','_partial')]]
    manifest_path=result/'merge_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text());manifest['combined_records']=len(canonical)
        manifest['complete_100_round_datasets_by_method']={m:sum(counts[d][m]==100 for d in datasets) for m in methods}
        manifest['live_TCLP_updates']='Additional computed TCLP seeds are recorded in provenance.jsonl'
        write(manifest_path,json.dumps(manifest,indent=2));names.append('merge_manifest.json')
    for name in names:
        destination=result/'all_methods'/name;temporary=destination.with_name(destination.name+'.tmp')
        shutil.copy2(result/name,temporary);temporary.replace(destination)
    status_path=result/'TCLP_resume_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    status.update(updated_at_utc=datetime.now(timezone.utc).isoformat(),total_successful_TCLP_rounds=sum(counts[d]['TCLP'] for d in datasets),remaining_successful_seeds=sum(100-counts[d]['TCLP'] for d in datasets),completed_datasets=sum(counts[d]['TCLP']==100 for d in datasets))
    write(status_path,json.dumps(status,indent=2))
    if added:print('PUBLISHED',[(r['dataset'],r['seed']) for r in added], 'remaining',status['remaining_successful_seeds'],flush=True)
