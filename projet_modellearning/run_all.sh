#!/bin/bash
# Full benchmark: every network, runs 1-10, with and without model learning (summary in <network>/summary.csv)
PY=${PY:-python}
for net in FN4 CN5 BN8 FN8; do
    $PY run_inference.py $net ${1:-1-10} > logs_${net}.log 2>&1
done
