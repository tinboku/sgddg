"""SFD-only intervention; fail closed on missing F2 caches or non-subset30 inputs."""
import argparse
import json
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, HERE, RESULTS_DIR, write_jsonl, sha256_text
from llm_cache import CachingClient, make_inner_client
from autoddg.description.search_locked import LockedSearchDescription, words
import autoddg.description.search_locked as implementation
import autoddg.description.fact_coverage as coverage


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', type=Path, required=True)
    ap.add_argument('--run-id', default='r1')
    ap.add_argument('--offline', action='store_true')
    args=ap.parse_args()
    cfg=yaml.safe_load(args.arm.read_text()); lock=cfg['locked_sfd']
    ref=yaml.safe_load((HERE/'configs/arm_F2.yaml').read_text())
    assert {k:v for k,v in cfg.items() if k not in ('arm','locked_sfd')} == {k:v for k,v in ref.items() if k!='arm'}
    ids=(HERE/'configs/subset30.txt').read_text().split()
    assert len(ids)==len(set(ids))==30
    rows=[]
    source_hash=sha256_text(Path(implementation.__file__).read_text()+Path(coverage.__file__).read_text()+Path(__file__).read_text())
    # Preflight every source before the first API call. Preserve existing failures.
    for i in ids:
        source=json.loads((CACHE_DIR/'gen'/lock['source_arm']/lock['source_run']/f'{i}.json').read_text())
        assert source['dataset_id']==i
        if source['status']=='ok':
            assert source.get('ufd') and source.get('sfd')
            prof=json.loads((HERE/source['profile_cache']).read_text())
            assert prof['profile_key']==source['profile_key']
            assert prof['table_sha256']==source['table_sha256']
            assert prof['factset']['budget']['selected_sfd']==source['selected_sfd']
    for i in ids:
        path=CACHE_DIR/'gen'/lock['source_arm']/lock['source_run']/f'{i}.json'
        source=json.loads(path.read_text())
        if source['status']!='ok':
            record={**source, 'arm':cfg['arm'], 'run_id':args.run_id,
                    'locked_source_status':source['status'], 'config':cfg}
            out=CACHE_DIR/'gen'/cfg['arm']/args.run_id/f'{i}.json'
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(json.dumps(record,ensure_ascii=False,indent=1))
            rows.append(record)
            print(f'{cfg["arm"]} {i} inherited {source["status"]}',flush=True)
            continue
        prof_path=HERE/source['profile_cache']; prof=json.loads(prof_path.read_text())
        assert prof['profile_key']==source['profile_key'] and prof['table_sha256']==source['table_sha256']
        assert prof['factset']['budget']['selected_sfd']==source['selected_sfd']
        fingerprint=sha256_text(path.read_text()+prof_path.read_text()+json.dumps(cfg,sort_keys=True)+source_hash)
        out=CACHE_DIR/'gen'/cfg['arm']/args.run_id/f'{i}.json'
        if out.exists() and (cached:=json.loads(out.read_text())).get('locked_fingerprint')==fingerprint:
            rows.append(cached); continue
        client=CachingClient(None if args.offline else make_inner_client('openai'), CACHE_DIR/'llm',
                             args.run_id, 'openai', offline=args.offline)
        engine=LockedSearchDescription(client,cfg['model'],cfg['sfd_temperature'])
        budget=words(source['sfd'])
        prompt,sfd=engine.generate(source['ufd'],source['topic'],source['sfd_facts'],budget)
        original=sfd; repairs=[]
        if lock['coverage_repair']:
            by_id={f['fact_id']:f for f in prof['factset']['facts']}
            sfd,repairs=engine.repair(source['ufd'],sfd,[by_id[k] for k in source['selected_sfd']],budget)
        record={**source,'arm':cfg['arm'],'run_id':args.run_id,'config':cfg,
                'config_hash':sha256_text(json.dumps(cfg,sort_keys=True)),
                'sfd':sfd,'sfd_prompt':prompt,'sfd_prompt_sha256':sha256_text(prompt),
                'calls':{'sfd':client.drain_log()}, 'locked_fingerprint':fingerprint,
                'locked_source':str(path.relative_to(HERE)), 'locked_source_sha256':sha256_text(path.read_text()),
                'locked_code_sha256':source_hash,'locked_budget_words':budget,
                'locked_pre_repair':original,'locked_repairs':repairs,
                'locked_coverage': ([{'id':f['fact_id'], 'before':coverage.fact_kept(f,original)[0],
                                     'after':coverage.fact_kept(f,sfd)[0]} for f in [by_id[k] for k in source['selected_sfd']]]
                                    if lock['coverage_repair'] else None)}
        assert words(sfd) <= budget + 50, f'SFD longer than matched budget: {words(sfd)} > {budget}'
        length_delta = words(sfd) - budget
        record={**source,'arm':cfg['arm'],'run_id':args.run_id,'config':cfg,
                'config_hash':sha256_text(json.dumps(cfg,sort_keys=True)),
                'sfd':sfd,'sfd_prompt':prompt,'sfd_prompt_sha256':sha256_text(prompt),
                'calls':{'sfd':client.drain_log()}, 'locked_fingerprint':fingerprint,
                'locked_source':str(path.relative_to(HERE)), 'locked_source_sha256':sha256_text(path.read_text()),
                'locked_code_sha256':source_hash,'locked_budget_words':budget,
                'locked_length_delta':length_delta,
                'locked_pre_repair':original,'locked_repairs':repairs,
                'locked_coverage': ([{'id':f['fact_id'], 'before':coverage.fact_kept(f,original)[0],
                                     'after':coverage.fact_kept(f,sfd)[0]} for f in [by_id[k] for k in source['selected_sfd']]]
                                    if lock['coverage_repair'] else None)}
        out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(record,ensure_ascii=False,indent=1))
        rows.append(record)
        print(f'{cfg["arm"]} {i} words={words(sfd)}/{budget} repairs={len(repairs)}',flush=True)
    write_jsonl(RESULTS_DIR/'descriptions'/f'{cfg["arm"]}__{args.run_id}__subset30.jsonl',rows)
    print(f'Completed {len(rows)} tables')

if __name__=='__main__': main()
