#!/bin/bash

# Usage: ./run.sh <input_dir> <split> <rate> <change> [mean_forcing] [stimulus] [prior]
#                 [force_basins] [temporal_basins] [ref] [test] [kov] [compute_proliferation]
#                 [use_proliferation] [--species human|mouse]
#
# Positional args 8-13 are optional flags (0/1):
#   ref                    : prepare_reference_network             (default 0)
#   test                   : infer_test + check_test                (default 0)
#   kov                    : simulate KOV + check KOV                (default 1)
#   compute_proliferation  : learn R_opt MLP, simulate with branching (default 0)
#   use_proliferation      : run get_proliferation_rates             (default 1)
#
# --species is a trailing named flag (default: human) — used by get_proliferation_rates
# for literature proliferation/death gene signatures.

# ./run.sh experimental_datasets/Semrau  full  0.7 0 1 1 1 1 1 0 0 1 0 0
# ./run.sh experimental_datasets/Kameneva  full  0.7 0 0.5 0.2 1 1 1 0 0 1 0 0 
./run.sh experimental_datasets/Schiebinger/ train 0.3 0 0.0 1 1 0 0 0 0 1 0 0 --species mouse

#  ./run.sh collaborations/orga_Olivier full 0.7 0 0.75 0.22 1 1 1 0 0 1 0 0 

# ./run.sh collaborations/Copycat_sc train 0.2 1 0.75 1 1 1 1 0 0 1 0 1 --species human
# ./run.sh collaborations/copycat/RMS2V3 train 0.6 0 0.5
# ./run.sh collaborations/copycat/RD136 train 0.8 0 0.5
# ./run.sh collaborations/copycat/RMS10 train 0.7 0 0.5
# ./run.sh collaborations/copycat/RMS_all train 0.5 0 0.5
