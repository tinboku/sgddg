"""Adversarial evidence checks and offline checks against the real subset30 seam."""
import json
from pathlib import Path
import pytest
from autoddg.description.query_expansion import (
    append_expansions, evidence_map, filter_entries, messages, validate_shape)

E = {'F1': 'Annual unemployment rate 2019 to 2023; county-level; Texas', 'column:Rate': 'Rate'}


def entry(text='unemployment rate', kind='measure', source='F1', semantic=None):
    parts = [{'text':text, 'kind':kind, 'sources':[source]}]
    if semantic:
        parts.append({'text':semantic, 'kind':'semantic', 'sources':[source]})
    return {'category':'topic_measure', 'parts':parts}


@pytest.mark.parametrize('text,kind', [('2024','year'), ('Paris','place'), ('wages','measure'),
    ('monthly','time'), ('19','number'), ('123','number')])
def test_unsupported_hard(text, kind):
    kept, rejected = filter_entries([entry(text,kind)], E)
    assert not kept and rejected[0]['reasons']


def test_unknown_source():
    assert filter_entries([entry(source='F999')], E)[1]


def test_numbers_hidden_in_semantics():
    assert filter_entries([entry(semantic='in 2024')], E)[1]


@pytest.mark.parametrize('semantic', ['yearly joblessness', 'by county', 'labor market conditions',
                                      'annual unemployment statistics'])
def test_semantic_association_is_open(semantic):
    assert filter_entries([entry(semantic=semantic)], E)[0]


def test_valid_hard_and_no_substring():
    assert filter_entries([entry('Texas','place')], E)[0]
    assert filter_entries([entry('2019','year')], E)[0]
    assert filter_entries([entry('employment','measure')], E)[1]


def test_metadata_not_evidence():
    ev = evidence_map([{'fact_id':'F1','columns':['Rate'],'render':{'ufd':'Rate annual'},
                       'priority':12345,'value':{'support_count':999}}], ['Rate'])
    assert ev == {'F1':'Rate\nRate annual', 'column:Rate':'Rate'}


def test_shape_and_budget():
    # Tolerant shape check: over-budget/invalid entries are dropped and reported,
    # not raised (a raise killed the whole 30-table run).
    entries, problems = validate_shape(json.dumps({'expressions':[entry()]*7}))
    assert len(entries) <= 6 and problems
    entries, problems = validate_shape(json.dumps({'expressions':[entry('word '*11)]}))
    assert entries == [] and 'expression_exceeds_ten_words' in problems
    entries, problems = validate_shape(json.dumps({'expressions':'not-a-list'}))
    assert entries == [] and problems == ['expressions_not_a_list']
    with pytest.raises(ValueError):
        append_expansions('base',[entry('word '*61)])
    assert append_expansions('base\n',[])[0] == 'base\n'


def test_real_subset30_immutable_and_no_query_input():
    b = Path(__file__).resolve().parents[1]
    ids=(b/'configs/subset30.txt').read_text().split()
    assert len(ids)==len(set(ids))==30
    ok=failed=0
    for did in ids:
        src=json.loads((b/f'cache/gen/bugfix/r1/{did}.json').read_text())
        if src['status']!='ok':
            failed+=1
            continue
        ok+=1
        f2=json.loads((b/f'cache/gen/F2/r1/{did}.json').read_text())
        prof=json.loads((b/f2['profile_cache']).read_text())
        assert prof['table_sha256']==src['table_sha256']
        byid={f['fact_id']:f for f in prof['factset']['facts']}
        ev=evidence_map([byid[i] for i in f2['selected_sfd']], [c['name'] for c in prof['factset']['columns']])
        prompt=messages(src['ufd'],src['topic'],ev)
        assert set(json.loads(prompt[-1]['content']))=={'UFD','topic','EVIDENCE'}
        for es in ([],[entry(semantic='yearly joblessness')]):
            out,tail=append_expansions(src['sfd'],es)
            assert out.encode()[:len(src['sfd'].encode())]==src['sfd'].encode()
            assert len(tail.split())<=60
    assert (ok,failed)==(29,1)
