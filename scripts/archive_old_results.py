#!/usr/bin/env python3
"""Archive unused, closed result directories; preview by default, never delete."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    results=ROOT/'results';results.mkdir(exist_ok=True)
    lock=(results/'.organization.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active=None
    state=ROOT/'run_state.json'
    if state.exists():active=Path(json.loads(state.read_text())['output']).resolve()
    references=[]
    for folder in ['docs','scripts','tests','config','test_data']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix in ['.md','.py','.sh','.json','.yaml','.txt','.cpp','.hpp']:
                references.append((str(path.relative_to(ROOT)),path.read_text(errors='replace')))
    for path in list(ROOT.glob('*.md'))+list(ROOT.glob('*.sh')):
        references.append((path.name,path.read_text(errors='replace')))
    open_paths=[]
    for process in Path('/proc').glob('[0-9]*'):
        for link in [process/'cwd',*list((process/'fd').glob('*'))]:
            try:
                target=Path(os.readlink(link).removesuffix(' (deleted)'))
                if target.is_absolute() and target.is_relative_to(results):open_paths.append(target)
            except OSError:pass
    plan=[];kept=[]
    for path in sorted(results.iterdir()):
        if not path.is_dir() or path.is_symlink() or path.name in ['history','organization']:continue
        reasons=[]
        if active and (active==path or active.is_relative_to(path)):reasons.append('current_run')
        if any(p==path or p.is_relative_to(path) for p in open_paths):reasons.append('open_by_process')
        reasons.extend('referenced_by:'+name for name,text in references if path.name in text)
        files=[p for p in path.rglob('*') if p.is_file() and not p.is_symlink()]
        if time.time()-max([path.stat().st_mtime,*[p.stat().st_mtime for p in files]])<600:
            reasons.append('modified_within_10_minutes')
        if reasons:kept.append(dict(path=path.name,reasons=reasons));continue
        match=re.search(r'(20\d{6})',path.name)
        date=datetime.strptime(match[1],'%Y%m%d').strftime('%Y-%m-%d') if match else 'undated'
        destination=results/'history'/date/path.name
        if destination.exists():raise RuntimeError('Archive target already exists: '+str(destination))
        plan.append(dict(source=str(path.relative_to(ROOT)),destination=str(destination.relative_to(ROOT)),
                         files=len(files),bytes=sum(p.stat().st_size for p in files)))
    report=dict(applied=args.apply,current_run=str(active) if active else None,planned=plan,kept=kept,moved=[])
    manifests=results/'organization';manifests.mkdir(exist_ok=True)
    manifest=manifests/(datetime.now().strftime('%Y%m%d_%H%M%S')+('_applied' if args.apply else '_preview')+'.json')
    manifest.write_text(json.dumps(report,indent=2))
    if args.apply:
        for item in plan:
            source=ROOT/item['source'];destination=ROOT/item['destination']
            if source.is_symlink() or not source.resolve().is_relative_to(results.resolve()):
                raise RuntimeError('Unexpected source path: '+str(source))
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.parent.resolve().is_relative_to(results.resolve()):
                raise RuntimeError('Archive escaped results directory')
            source.rename(destination)
            report['moved'].append(item)
            manifest.write_text(json.dumps(report,indent=2))
        if active and active.is_relative_to(results):
            latest=results/'latest'
            if latest.exists() and not latest.is_symlink():raise RuntimeError('results/latest is not a symlink')
            temporary=results/('.latest_'+str(os.getpid()))
            temporary.symlink_to(active.relative_to(results),target_is_directory=True);temporary.replace(latest)
    print(json.dumps(dict(applied=args.apply,directories=len(plan),archived_bytes=sum(x['bytes'] for x in plan),
                          preserved_directories=len(kept),manifest=str(manifest)),indent=2))


if __name__=='__main__':main()
