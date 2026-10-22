#!/bin/zsh
set -e
source ~/.zshrc >/dev/null 2>&1
set -e
cd "${0:A:h}"
PY=../../.venvs/autoddg-deep/bin/python
$PY -I 01b_confusion_pairs.py
$PY -I 02_generate_discriminative.py "$@"
$PY -I 03_index_eval.py --arms baseline__r1__subset30,baseline__r2__subset30,bugfix__r1__subset30,F2__r1__subset30,D1__r1__subset30 --views sfd --retrievers bm25 --pools per_query --dataset-ids configs/subset30.txt --tag d1
for pair in 'bugfix D1' 'F2 D1' 'baseline baseline'; do
  parts=(${=pair})
  second=r1
  if [[ $parts[1] == baseline ]]; then second=r2; fi
  $PY -I 04_paired_test.py --eval results/eval/d1.json --a ${parts[1]}__r1__subset30 --b ${parts[2]}__${second}__subset30
done
$PY -I 06_discrimination.py
