from pathlib import Path
import hashlib, json
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]; BASE=ROOT/'experiments/autoddg_deep'; OUT=BASE/'adaptive_preval'; OUT.mkdir(parents=True,exist_ok=True)
SUB=BASE/'configs/subset30.txt'; DATA=BASE/'data/tables'
def feats(i):
    try: d=pd.read_csv(DATA/f'{i}.csv', low_memory=False)
    except (UnicodeDecodeError,pd.errors.ParserError): d=pd.read_csv(DATA/f'{i}.csv', encoding='latin-1', engine='python', on_bad_lines='skip')
    n=len(d); miss=d.isna().mean(); cats=[c for c in d if d[c].dtype=='object' and d[c].nunique(dropna=True)<10]
    return {'id':i,'rows':n,'cols':len(d.columns),'missing':float(miss.max()),'cat_repeat':bool(n>1000 and any(d[c].nunique(dropna=True)<10 for c in cats))}
def main():
    rows=[feats(x.strip()) for x in SUB.read_text().splitlines() if x.strip()]
    used=set(); selected=[]
    rules=[('wide',lambda x:x['cols']>=40,2),('small',lambda x:x['rows']<=100,3),('medium',lambda x:100<x['rows']<10000,3),('large',lambda x:x['rows']>10000,3),('missing',lambda x:x['missing']>0.30,2),('categorical_repeat',lambda x:x['cat_repeat'],1)]
    for name,pred,k in rules:
        got=[x for x in rows if x['id'] not in used and pred(x)][:k]
        if len(got)!=k: raise SystemExit(f'{name}: need {k}, got {len(got)}')
        for x in got: x['group']=name; selected.append(x); used.add(x['id'])
    text='\n'.join(x['id'] for x in selected)+'\n'; (OUT/'configs').mkdir(exist_ok=True); (OUT/'configs/adaptive_preval_tables.txt').write_text(text)
    (OUT/'configs/adaptive_preval_tables.sha256').write_text(hashlib.sha256(text.encode()).hexdigest()+'  adaptive_preval_tables.txt\n')
    (OUT/'selection.json').write_text(json.dumps({'rules':[x[0] for x in rules],'tables':selected},indent=2)+'\n')
if __name__=='__main__': main()
