"""Q0/Q1 share one cached request per successful subset30 table; immutable bugfix SFD."""
import argparse
import json
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, CACHE_DIR, RESULTS_DIR, sha256_text, write_jsonl
from llm_cache import CachingClient, make_inner_client
from autoddg.description import query_expansion as q


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--offline', action='store_true')
    args = ap.parse_args()
    ids = (HERE/'configs/subset30.txt').read_text().split()
    assert len(ids) == len(set(ids)) == 30
    configs = {a: yaml.safe_load((HERE/f'configs/arm_{a}.yaml').read_text()) for a in ('Q0','Q1')}
    reference = yaml.safe_load((HERE/'configs/arm_bugfix.yaml').read_text())
    for arm, cfg in configs.items():
        assert {k:v for k,v in cfg.items() if k not in ('arm','query_expansion')} == {k:v for k,v in reference.items() if k != 'arm'}
        assert cfg['query_expansion'] == dict(source_arm='bugfix',source_run='r1',fact_source='F2/r1',temperature=0.0,max_expressions=6,max_words=60,filter_evidence=(arm=='Q1'))
    inputs = load_inputs()
    client = CachingClient(None if args.offline else make_inner_client('openai'), CACHE_DIR/'llm', 'q_r1', 'openai', offline=args.offline)
    rows = {a: [] for a in configs}
    generate(inputs, client, configs, rows)


def load_inputs():
    ids = (HERE/'configs/subset30.txt').read_text().split()
    assert len(ids) == len(set(ids)) == 30
    inputs = []
    # All input validation before any API request; no raw table reads/profiling.
    for did in ids:
        src = json.loads((CACHE_DIR/f'gen/bugfix/r1/{did}.json').read_text())
        assert src['dataset_id'] == did
        ev = {}
        if src['status'] == 'ok':
            f2 = json.loads((CACHE_DIR/f'gen/F2/r1/{did}.json').read_text())
            prof = json.loads((HERE/f2['profile_cache']).read_text())
            assert src['table_sha256'] == f2['table_sha256'] == prof['table_sha256']
            assert prof['profile_key'] == f2['profile_key']
            assert prof['factset']['budget']['selected_sfd'] == f2['selected_sfd']
            byid = {f['fact_id']: f for f in prof['factset']['facts']}
            ev = q.evidence_map([byid[i] for i in f2['selected_sfd']], [c['name'] for c in prof['factset']['columns']])
        inputs.append((did,src,ev))
    return inputs


def generate(inputs, client, configs, rows):
    for did,src,ev in inputs:
        audit = {'dataset_id':did, 'source_status':src['status']}
        entries, kept = [], []
        if src['status'] == 'ok':
            msg = q.messages(src['ufd'], src['topic'], ev)
            response = client.chat_completions_create(model='gpt-4o-mini',messages=msg,temperature=0.0,
                response_format={'type':'json_object'}, max_tokens=1800)
            entries, shape_problems = q.validate_shape(response['choices'][0]['message']['content'])
            kept, rejected = q.filter_entries(entries, ev)
            audit.update(entries=entries, shape_problems=shape_problems, kept=kept, rejected=rejected, before=len(entries), after=len(kept),
                         evidence=ev, calls=client.drain_log(), source_sfd_sha256=sha256_text(src['sfd']),
                         request_sha256=sha256_text(json.dumps(msg,sort_keys=True)),
                         implementation_sha256=sha256_text(Path(q.__file__).read_text()))
        for arm in configs:
            record = {**src, 'arm':arm, 'config':configs[arm], 'calls':{}, 'query_expansion':audit}
            if src['status'] == 'ok':
                record['sfd'],tail = q.append_expansions(src['sfd'], kept if arm == 'Q1' else entries)
                record['query_expansion'] = {**audit, 'suffix':tail, 'suffix_words':len(tail.split())}
                assert record['sfd'] == src['sfd'] + ('\n\n'+tail if tail else '')
            out = CACHE_DIR/f'gen/{arm}/r1/{did}.json'
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(json.dumps(record,ensure_ascii=False,indent=1))
            rows[arm].append(record)
        print(f'{did} status={src["status"]} expansions={len(entries)} -> {len(kept)}',flush=True)
    for arm, records in rows.items():
        write_jsonl(RESULTS_DIR/f'descriptions/{arm}__r1__subset30.jsonl',records)

if __name__ == '__main__':
    main()
