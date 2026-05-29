#!/bin/bash

# Usage: ./run.sh <input_dir> <split> <change> <rate> [mean_forcing] [stimulus] [prior]
#                 [force_basins] [temporal_basins] [rd] [ref] [test] [kov]
#
# Positional args 5-13 are optional flags (0/1):
#   rd   : read-depth correction         (default 0)
#   ref  : prepare_reference_network     (default 0)
#   test : infer_test + check_test       (default 0)
#   kov  : simulate KOV + check KOV      (default 1)

# ./run.sh experimentalxsdatasets/Kameneva  full  0 0.7 0.5 
./run.sh experimental_datasets/Schiebinger/ train 0 0.3 0 1 1 0 0 0 0 0 1

#  ./run.sh collaborations/orga_Olivier full 0 1.0 0.5
# ./run.sh collaborations/copycat/RMS2V3 train 1 0.6 0.5
# ./run.sh collaborations/copycat/RD136 train 1 0.8 0.5
# ./run.sh collaborations/copycat/RMS10 train 1 0.7 0.5
# ./run.sh collaborations/copycat/RMS_all train 1 0.5 0.5
