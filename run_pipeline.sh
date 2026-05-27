#!/bin/bash

# Usage: ./run.sh <input_dir> <split> <change> <rate> <mean> [stimulus=1.0] [prior=1.0]
#   stimulus      : float in [0,1] — penalize stimulus edges (0=no stimulus, 1=full stimulus)
#   prior         : float in [0,1] — penalize edges absent from prior network (0=impossible, 1=no prior)
# Omitting the last two arguments uses the defaults defined in NetworkModel (1.0 for both).

./run.sh experimental_datasets/Semrau full 0 0.7 1.0
./run.sh experimental_datasets/Kameneva full 0 0.7 0.5
./run.sh experimental_datasets/Schiebinger/ train 0 0.3 0.5

#  ./run.sh collaborations/orga_Olivier full 0 1.0 0.5
# ./run.sh collaborations/copycat/RMS2V3 train 1 0.6 0.5
# ./run.sh collaborations/copycat/RD136 train 1 0.8 0.5
# ./run.sh collaborations/copycat/RMS10 train 1 0.7 0.5
# ./run.sh collaborations/copycat/RMS_all train 1 0.5 0.5


