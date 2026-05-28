#!/bin/bash

# Usage: ./run.sh <input_dir> <split> <change> <rate> <mean> [stimulus] [prior] [force_basins] [temporal_basins]
#   stimulus       : float in [0,1] — penalize stimulus edges (0=no stimulus, 1=full stimulus)
#   prior          : float in [0,1] — penalize edges absent from prior network (0=impossible, 1=no prior)
#   force_basins   : float in [0,1] — preserve mode means in NB mixture
#   temporal_basins: 0 or 1 — preserve mode means temporally
# Omitting optional arguments uses the defaults defined in NetworkModel (base.py).

./run.sh experimental_datasets/Semrau full 0 0.7 1.0 1.0 1.0 1.0 1
./run.sh experimental_datasets/Kameneva full 0 0.7 0.5 1.0 1.0 1.0 1
./run.sh experimental_datasets/Schiebinger/ train 0 0.3 0.5 1.0 1.0 0.0 0

#  ./run.sh collaborations/orga_Olivier full 0 1.0 0.5
# ./run.sh collaborations/copycat/RMS2V3 train 1 0.6 0.5
# ./run.sh collaborations/copycat/RD136 train 1 0.8 0.5
# ./run.sh collaborations/copycat/RMS10 train 1 0.7 0.5
# ./run.sh collaborations/copycat/RMS_all train 1 0.5 0.5


