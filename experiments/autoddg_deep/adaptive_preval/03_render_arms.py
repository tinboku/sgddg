from pathlib import Path
import json, sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; BASE=ROOT/'experiments/autoddg_deep'; RUN=BASE/'adaptive_preval'; DATA=BASE/'data/tables'; OUT=RUN/'results'; OUT.mkdir(exist_ok=True)
T='TABLE {id}\nROWS {rows} COLS {cols}\n{body}\nSAMPLE\n{sample}\nRELATION {rel}'
def main():
 for ident in (RUN/'configs/adaptive_preval_tables.txt').read_text().splitlines():
  try: d=pd.read_csv(DATA/f'{ident}.csv',low_memory=False)
  except (UnicodeDecodeError,pd.errors.ParserError): d=pd.read_csv(DATA/f'{ident}.csv',encoding='latin-1',engine='python',on_bad_lines='skip')
  gt=json.loads((RUN/'ground_truth'/f'{ident}.json').read_text()); lines=[]
  for c in gt['columns']:
   x=f"{c['name']} dtype={c['dtype']} missing={c['missing_rate']:.6f} unique={c['unique']}"
   if 'min' in c:x+=f" min={c['min']} max={c['max']}"
   if 'top3' in c:x+=f" top3={c['top3']}"
   lines.append(x)
  p1=T.format(id=ident,rows=len(d),cols=len(d.columns),body='\n'.join(lines),sample=d.head(5).to_csv(index=False),rel='none')
  p2=T.format(id=ident,rows=len(d),cols=len(d.columns),body='\n'.join(lines)+'\nQUANTILES\n'+d.select_dtypes('number').quantile([.05,.25,.5,.75,.95]).to_string(),sample=d.head(5).to_csv(index=False),rel=str(gt['relationships'][0] if gt['relationships'] else 'none'))
  sys.path.insert(0,str(ROOT/'external/AutoDDG/src'))
  try:
   from autoddg.profiling import profile_dataset; sample=d.head(100); prof,sem=profile_dataset(sample); p0=T.format(id=ident,rows=len(d),cols=len(d.columns),body=prof+'\n'+sem,sample=sample.to_csv(index=False),rel='profiler output')
  except Exception as e: p0='P0_ERROR '+repr(e)
  trunc=abs(len(p2)-len(p1))/max(1,len(p1))>.15
  if trunc:p2=p2[:int(len(p1)*1.15)]
  for a,s in (('P0',p0),('P1',p1),('P2',p2)): (OUT/f'{ident}_{a}.txt').write_text(s)
  (OUT/f'{ident}.json').write_text(json.dumps({'table':ident,'lengths':{a:len((OUT/f'{ident}_{a}.txt').read_text()) for a in ('P0','P1','P2')},'truncated':trunc},indent=2)+'\n')
if __name__=='__main__': main()
