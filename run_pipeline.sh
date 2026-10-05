#!/bin/bash

# Usage: ./run.sh <input_dir> [stimulus] [prior] [mean_forcing] [force_basins] [temporal_basins]
# (empty or -1 = value of the Model_parameters sheet of Data/CardamomOT_inputs.xlsx, else default)
#
# The other settings of a project are parameters of its Model_parameters sheet, e.g. for the examples
# below (former positional arguments):
#   Semrau       : split=full,  train_rate=0.7, select_genes=False, simulate_perturbations=True
#   Kameneva     : split=full,  train_rate=0.7, select_genes=False, simulate_perturbations=True
#   Schiebinger  : split=train, train_rate=0.2, select_genes=False, simulate_perturbations=True, species=mouse
#   Copycat_sc   : split=train, train_rate=0.1, select_genes=True, estimate_proliferation_rates=True,
#                  simulate_with_proliferation=True, simulate_perturbations=True, species=human

# ./run.sh experimental_datasets/Semrau 1 1 1 1 1
# ./run.sh experimental_datasets/Kameneva 0.2 1 0.75 1 1
./run.sh experimental_datasets/Schiebinger 1 1 0.5 1 1

# ./run.sh collaborations/Copycat_sc 1 0 1 1 1
