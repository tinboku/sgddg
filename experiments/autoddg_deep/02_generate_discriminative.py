"""D1: discriminative SFD on subset30; F2 profile/facts/topic/UFD reused, one SFD call per table.

Fails closed before the first API call if an F2 cache, its profile or the
neighbour list is missing or inconsistent. F2 failures are inherited. The SFD
is cut to the F2 SFD word count; a shorter output is kept and recorded as
discriminative_length_delta (< 0).
"""
import argparse
import json
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, HERE, RESULTS_DIR, write_jsonl, sha256_text, select_benchmark
from llm_cache import CachingClient, make_inner_client
from autoddg.description.search_discriminative import DiscriminativeSearchDescription, words
import autoddg.description.search_discriminative as implementation


def neighbour_text(nid, corpus, src_arm, src_run):
    rec = json.loads((CACHE_DIR/'gen'/src_arm/src_run/f'{nid}.json').read_text())
    title = corpus[nid].get('title') or ''
    if rec['status'] == 'ok':
        text = f"Title: {title}\nDescription: {rec['ufd']}"
        if rec.get('sfd_facts'):
            text += '\nComputed facts:\n' + rec['sfd_facts']
        return text
    return f"Title: {title}\nDescription: {corpus[nid].get('description') or ''}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', type=Path, default=HERE/'configs/arm_D1.yaml')
    ap.add_argument('--run-id', default='r1')
    ap.add_argument('--offline', action='store_true')
    args = ap.parse_args()
    cfg = yaml.safe_load(args.arm.read_text()); disc = cfg['discriminative_sfd']
    ref = yaml.safe_load((HERE/'configs/arm_F2.yaml').read_text())
    assert {k: v for k, v in cfg.items() if k not in ('arm', 'discriminative_sfd')} == {k: v for k, v in ref.items() if k != 'arm'}
    ids = (HERE/'configs/subset30.txt').read_text().split()
    assert len(ids) == len(set(ids)) == 30
    pairs_path = HERE/disc['neighbours']
    neighbours = json.loads(pairs_path.read_text())['neighbours']
    assert set(neighbours) == set(ids)
    _, _, corpus = select_benchmark(csv_only=True)
    src_dir = CACHE_DIR/'gen'/disc['source_arm']/disc['source_run']
    code_hash = sha256_text(Path(implementation.__file__).read_text() + Path(__file__).read_text())
    for i in ids:  # preflight
        source = json.loads((src_dir/f'{i}.json').read_text())
        assert source['dataset_id'] == i and neighbours[i]['neighbour'] in ids
        if source['status'] == 'ok':
            prof = json.loads((HERE/source['profile_cache']).read_text())
            assert prof['profile_key'] == source['profile_key'] and prof['table_sha256'] == source['table_sha256']
            assert prof['factset']['budget']['selected_sfd'] == source['selected_sfd']
    client = CachingClient(None if args.offline else make_inner_client('openai'), CACHE_DIR/'llm',
                           args.run_id, 'openai', offline=args.offline)
    engine = DiscriminativeSearchDescription(client, cfg['model'], cfg['sfd_temperature'])
    rows = []
    for i in ids:
        path = src_dir/f'{i}.json'
        source = json.loads(path.read_text())
        out = CACHE_DIR/'gen'/cfg['arm']/args.run_id/f'{i}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        nb = neighbours[i]
        if source['status'] != 'ok':
            record = {**source, 'arm': cfg['arm'], 'run_id': args.run_id, 'config': cfg,
                      'discriminative_source_status': source['status'], 'discriminative_neighbour': nb}
            out.write_text(json.dumps(record, ensure_ascii=False, indent=1)); rows.append(record)
            print(f'{cfg["arm"]} {i} inherited {source["status"]}', flush=True)
            continue
        ntext = neighbour_text(nb['neighbour'], corpus, disc['source_arm'], disc['source_run'])
        budget = words(source['sfd'])
        # Guard: without the neighbour insertions the prompt must be F2's prompt.
        assert engine.build_prompt(source['ufd'], source['topic'], source['sfd_facts']) == source['sfd_prompt']
        prompt, sfd, raw = engine.generate(source['ufd'], source['topic'], source['sfd_facts'], ntext, budget)
        record = {**source, 'arm': cfg['arm'], 'run_id': args.run_id, 'config': cfg,
                  'config_hash': sha256_text(json.dumps(cfg, sort_keys=True)),
                  'sfd': sfd, 'sfd_prompt': prompt, 'sfd_prompt_sha256': sha256_text(prompt),
                  'calls': {'sfd': client.drain_log()},
                  'discriminative_source': str(path.relative_to(HERE)),
                  'discriminative_source_sha256': sha256_text(path.read_text()),
                  'discriminative_code_sha256': code_hash,
                  'discriminative_neighbour': nb, 'discriminative_neighbour_text': ntext,
                  'discriminative_raw': raw, 'discriminative_budget_words': budget,
                  'discriminative_length_delta': words(sfd) - budget}
        out.write_text(json.dumps(record, ensure_ascii=False, indent=1)); rows.append(record)
        print(f'{cfg["arm"]} {i} neighbour={nb["neighbour"][:8]} tier={nb["tier"]} '
              f'words={words(sfd)}/{budget} raw={words(raw)}', flush=True)
    write_jsonl(RESULTS_DIR/'descriptions'/f'{cfg["arm"]}__{args.run_id}__subset30.jsonl', rows)
    print(f'Completed {len(rows)} tables')


if __name__ == '__main__':
    main()
