#!/bin/bash

# Usage: ./run.sh <input_dir> <split> <rate> <change> [mean_forcing] [stimulus]
#                 [force_basins] [temporal_basins] [ref] [prior] [test] [kov] [simulate_proliferation]
#                 [use_proliferation] [--species human|mouse]
#
#   ref                    : build_reference_network (0/1, default 0)
#   prior                  : weight of edges absent from the prior (0 = hard/sparse, 1 = ignored, -1 = model default)
#   test                   : infer_test + check_test (0/1, default 0)
#   kov                    : simulate KOV + check KOV (0/1, default 1)
#   simulate_proliferation : simulate with proliferation/death (0/1, default 0)
#   use_proliferation      : run get_proliferation_rates (0/1, default 0)
#
# --species is a trailing named flag (human|mouse) — used by get_proliferation_rates
# (proliferation/death gene signatures) and get_degradation_rates (half-life tables).
# If omitted, both detect the species from gene names.

# ./run.sh experimental_datasets/Semrau  full  0.7 0 1 1 1 1 0 1 0 1 0 0
# ./run.sh experimental_datasets/Kameneva  full  0.7 0 0.5 0.2 1 1 0 1 0 1 0 0 
# ./run.sh experimental_datasets/Schiebinger/ train 0.2 0 0.5 1 1 1 0 1 0 1 0 0 --species mouse

./run.sh collaborations/Copycat_sc train 0.1 1 0.75 1 1 1 1 0 0 1 0 1 --species human
