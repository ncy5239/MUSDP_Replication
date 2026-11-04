"""Resume the repository's original TCLP defaults over all datasets.

Run from an extracted MUSDP_Replication-main folder (or put this script there):
    python run_TCLP_original_to_completion.py

This is a synchronous, locally run program. It never fabricates missing rounds.
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Resume 100 bootstrap attempts for TCLP per dataset using the repository original defaults.')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1], help='Folder containing data/ and test/')
    args = parser.parse_args()
    repo = args.repo.resolve()
    runner = repo/'test'/'demo_all_methods.py'
    data = repo/'data'
    result = repo/'result'
    if not runner.exists() or not data.exists():
        raise SystemExit('Cannot locate repository test/demo_all_methods.py and data/. Supply --repo PATH')
    dependencies = repo.parent/'.test-deps'
    if dependencies.exists():
        sys.path.insert(0, str(dependencies))
    sys.path.insert(0, str(repo))
    from utilities.publish_tclp import publish, read_rows, write
    spec = importlib.util.spec_from_file_location('team_experiment',runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    datasets = sorted(p.stem for p in data.glob('*.csv'))
    if not datasets:
        raise SystemExit('No CSV dataset files in '+str(data))
    lock=result/'TCLP_resume.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise SystemExit('A TCLP resume worker already holds '+str(lock))
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    status_path=result/'TCLP_resume_status.json'
    def status(**updates):
        value=json.loads(status_path.read_text()) if status_path.exists() else {}
        value.update(updates);write(status_path,json.dumps(value,indent=2))
    try:
        backup=repo.parent/'tmp'/('before_TCLP_resume_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        backup.mkdir(parents=True,exist_ok=False)
        for f in result.iterdir():
            if f.is_file() and f.suffix in ('.csv','.json','.jsonl','.md'):shutil.copy2(f,backup/f.name)
        status(status='running',pid=os.getpid(),backup=str(backup),settings=dict(n_estimators=500,cv_folds=10,tune_mtry=True))
        # Recover successful batch records from earlier interrupted invocations.
        for batch in sorted(result.glob('TCLP_original_*')):
            if batch.is_dir() and (batch/'rounds.jsonl').exists():publish(result,batch)
        for dataset in datasets:
            successful={int(r['seed']) for r in read_rows(result/'rounds.jsonl') if r['dataset']==dataset and r['method']=='TCLP'}
            missing=sorted(set(range(100))-successful)
            if not missing:
                print('SKIP 100 successful seeds:',dataset,flush=True);continue
            groups=[]
            for seed in missing:
                if groups and seed==groups[-1][-1]+1:groups[-1].append(seed)
                else:groups.append([seed])
            for group in groups:
                output=result/f'TCLP_original_{dataset}_seed{group[0]:03d}'
                suffix=1
                while output.exists():
                    output=result/f'TCLP_original_{dataset}_seed{group[0]:03d}_{suffix}';suffix+=1
                status(dataset=dataset,first_seed=group[0],last_seed=group[-1],batch=str(output))
                print('RUN',dataset,'seeds',group[0],'through',group[-1],flush=True)
                module.run_all_methods(data_dir=data,save_path=output,datasets=[dataset],methods=['TCLP'],
                                       Reps=len(group),base_seed=group[0],n_estimators=500,cv_folds=10,
                                       tune_mtry=True,checkpoint_callback=lambda batch:publish(result,batch))
        counts=json.loads((result/'successful_rounds.json').read_text())
        remaining=sum(100-counts[d]['TCLP'] for d in datasets)
        status(status='complete' if remaining==0 else 'finished_with_missing_successes',remaining_successful_seeds=remaining)
        print('FINISHED; missing successful seeds:',remaining,flush=True)
    except BaseException as error:
        status(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error));raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
