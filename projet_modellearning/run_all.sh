#!/bin/bash
# Full benchmark: every network, runs 1-10, observed / hidden proteins, with and without model learning
# (summary in bench/<network>/summary.csv)
PY=${PY:-python}
for net in FN4 CN5 BN8 FN8; do
    $PY run_inference.py $net ${1:-1-10} > bench/logs_${net}.log 2>&1
done
