#!/bin/zsh
set -e
source ~/.zshrc >/dev/null 2>&1
set -e
cd "${0:A:h}"
PY=../../.venvs/autoddg-deep/bin/python
$PY -I 02_generate_locked.py --arm configs/arm_L1.yaml --run-id r1
# The identical supplement request must hit L1's cache; L2 repair is deterministic.
$PY -I 02_generate_locked.py --arm configs/arm_L2.yaml --run-id r1 --offline
$PY -I 03_index_eval.py --arms baseline__r1__subset30,baseline__r2__subset30,bugfix__r1__subset30,F2__r1__subset30,L1__r1__subset30,L2__r1__subset30 --views sfd --retrievers bm25 --pools per_query --dataset-ids configs/subset30.txt --tag l1l2
for pair in 'bugfix L2' 'F2 L2' 'F2 L1' 'L1 L2' 'baseline baseline'; do
  parts=(${=pair})
  second=r1
  if [[ $parts[1] == baseline ]]; then second=r2; fi
  $PY -I 04_paired_test.py --eval results/eval/l1l2.json --a ${parts[1]}__r1__subset30 --b ${parts[2]}__${second}__subset30
 done
$PY -I 05_fact_audit.py --arms F2,L1,L2 --dataset-ids configs/subset30.txt
$PY -I 06_report_locked.py
