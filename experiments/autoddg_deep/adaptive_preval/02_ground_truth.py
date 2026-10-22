from pathlib import Path
import hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; BASE=ROOT/'experiments/autoddg_deep'; RUN=BASE/'adaptive_preval'; DATA=BASE/'data/tables'; OUT=RUN/'ground_truth'; OUT.mkdir(parents=True,exist_ok=True)
def jsonable(x):
    if pd.isna(x): return None
    return x.item() if hasattr(x,'item') else str(x)
def main():
  for ident in (RUN/'configs/adaptive_preval_tables.txt').read_text().splitlines():
    try: d=pd.read_csv(DATA/f'{ident}.csv',low_memory=False)
    except (UnicodeDecodeError,pd.errors.ParserError): d=pd.read_csv(DATA/f'{ident}.csv',encoding='latin-1',engine='python',on_bad_lines='skip')
    cols=[]; dates=[]
    for c in d:
      s=d[c]; dtype=str(s.dtype); q={'name':c,'dtype':dtype,'missing_rate':float(s.isna().mean()),'unique':int(s.nunique(dropna=True))}
      if pd.api.types.is_numeric_dtype(s): q['min']=jsonable(s.min()); q['max']=jsonable(s.max())
      if s.dtype=='object': q['top3']={str(k):int(v) for k,v in s.value_counts(dropna=True).head(3).items()}
      dt=pd.to_datetime(s,errors='coerce');
      if dt.notna().mean()>.8 and dt.notna().any(): q['time_range']=[str(dt.min()),str(dt.max())]; dates.append(c)
      if q['unique']==len(d) and len(d)>0: q['primary_key_candidate']=True
      cols.append(q)
    rel=[]
    for i,a in enumerate(d.columns):
      for b in d.columns[i+1:]:
       if d[a].equals(d[b]): rel.append({'type':'equal','left':a,'right':b})
       elif pd.api.types.is_numeric_dtype(d[a]) and pd.api.types.is_numeric_dtype(d[b]):
        z=d[[a,b]].dropna();
        if len(z) and (z[a]>=z[b]).mean()>=.99: rel.append({'type':'ge','left':a,'right':b})
       if len(rel)>=2: break
      if len(rel)>=2: break
    if len(rel)<2 and len(d.columns)>=2: rel += [{'type':'distinct_pair','left':d.columns[0],'right':d.columns[1]}]
    obj={'table':ident,'rows':len(d),'cols':len(d.columns),'columns':cols,'relationships':rel[:2],'facts_count':sum(1 for c in cols for k in c if k not in ('name','top3'))+len(rel[:2])}
    p=OUT/f'{ident}.json'; p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n'); (OUT/f'{ident}.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
if __name__=='__main__': main()
