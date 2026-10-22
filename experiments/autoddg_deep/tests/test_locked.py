import pytest
from autoddg.description.search_locked import LockedSearchDescription, words
from autoddg.description.fact_coverage import fact_kept


def fact(i, value):
    return {'fact_id':i,'kind':'temporal_span','value':{'start':value,'end':value},
            'render':{'ufd':'Coverage: '+value}}


def test_normalized_quarter():
    assert fact_kept(fact('F1','2023-Q1'),'2023 Q1')[0]


def test_repair_trim_cascade_and_immutable_overview():
    ufd='Verbatim  spacing\nand newline.'
    sfd='Dataset Overview:\n'+ufd+'\n\n'+'keywords '*20+'2020'
    fixed, repairs=LockedSearchDescription.repair(ufd,sfd,[fact('F1','2021'),fact('F2','2020')],words(sfd))
    assert fixed.startswith('Dataset Overview:\n'+ufd+'\n\n')
    assert len(repairs)==2
    assert all(fact_kept(fact('F',y),fixed)[0] for y in ('2020','2021'))
    assert words(fixed)<=words(sfd)
    again, repairs2=LockedSearchDescription.repair(ufd,fixed,[fact('F1','2021')],words(sfd))
    assert again==fixed and repairs2==[]


def test_impossible_budget_fails():
    with pytest.raises(ValueError,match='exceed'):
        LockedSearchDescription.repair('text','Dataset Overview:\ntext\n\nmore',[fact('F1','2021')],3)


def test_all_subset30_cached_inputs_fit_repair_budget():
    """Offline fixture verification, never creates experimental L1/L2 outputs."""
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    ids=(root/'configs/subset30.txt').read_text().split()
    assert len(ids)==30
    for i in ids:
        path=root/'cache/gen/F2/r1'/f'{i}.json'
        if not path.exists():
            pytest.skip('Local F2 cache unavailable')
        rec=json.loads(path.read_text())
        if rec['status']!='ok':
            continue
        prof=json.loads((root/rec['profile_cache']).read_text())
        facts={f['fact_id']:f for f in prof['factset']['facts']}
        selected=[facts[k] for k in rec['selected_sfd']]
        sfd='Dataset Overview:\n'+rec['ufd']+'\n\nKeywords: data'
        result,_=LockedSearchDescription.repair(rec['ufd'],sfd,selected,words(rec['sfd']))
        assert words(result)<=words(rec['sfd'])
        assert all(fact_kept(f,result)[0] for f in selected)


def test_generate_exact_length_and_verbatim_overview():
    from autoddg.llm import LLMClient
    class Stub(LLMClient):
        def chat_completions_create(self, model, messages, temperature=0.0, **kwargs):
            assert 'full-table facts' in messages[-1]['content']
            return {'choices':[{'message':{'content':'Themes: '+ 'context '*100}}]}
    engine=LockedSearchDescription(Stub())
    ufd='Keep  whitespace\nand punctuation.'
    _, result=engine.generate(ufd,'topic','full-table facts',40)
    assert result.startswith('Dataset Overview:\n'+ufd+'\n\n')
    assert words(result)==40


def test_short_generation_kept_with_length_confound():
    # New behavior: a short supplement is kept as-is (no padding, no raise);
    # the runner records the unequal length as locked_length_delta and the
    # report discloses it as a confound. A raise killed the whole run.
    from autoddg.llm import LLMClient
    class Stub(LLMClient):
        def chat_completions_create(self, **kwargs):
            return {'choices':[{'message':{'content':'short'}}]}
    _, result = LockedSearchDescription(Stub()).generate('ufd','topic','facts',40)
    assert result.endswith('short')
    assert 'ufd' in result
