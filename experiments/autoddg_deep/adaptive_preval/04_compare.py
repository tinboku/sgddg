from pathlib import Path
import json,math
ROOT=Path(__file__).resolve().parents[3]; RUN=ROOT/'experiments/autoddg_deep/adaptive_preval'; OUT=RUN/'results'
def main():
 allr=[]
 for ident in (RUN/'configs/adaptive_preval_tables.txt').read_text().splitlines():
  gt=json.loads((RUN/'ground_truth'/f'{ident}.json').read_text()); row={'table':ident}
  for arm in ('P0','P1','P2'):
   s=(OUT/f'{ident}_{arm}.txt').read_text(); ok=miss=dist=0
   for c in gt['columns']:
    present=c['name'] in s; ok+=int(present); miss+=int(not present); dist+=int(present and 'min' in c and f"min={c['min']}" not in s)
   row[arm]={'correct':ok,'omitted':miss,'distorted':dist,'relation_retained':int(any(str(x) in s for x in gt['relationships'])),'chars':len(s),'tokens':math.ceil(len(s)/4)}
  allr.append(row)
 totals={a:{k:sum(x[a][k] for x in allr) for k in ('correct','omitted','distorted','relation_retained','chars','tokens')} for a in ('P0','P1','P2')}
 (OUT/'comparison.json').write_text(json.dumps({'tables':allr,'totals':totals},indent=2)+'\n')
if __name__=='__main__': main()
