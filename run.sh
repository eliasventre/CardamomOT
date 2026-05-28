#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cardamom_light

# Usage: ./run.sh <input_dir> <split> <change> <rate> <mean> [stimulus] [prior]
#                 [force_basins] [temporal_basins] [rd] [ref] [test] [kov]
#
#   split          : full | train
#   change         : 0/1 — differential gene selection
#   rate           : float — rate parameter for kinetics
#   mean           : float — mean expression threshold (-1 = auto)
#   stimulus       : float in [0,1] — penalize stimulus edges (-1 = model default)
#   prior          : float in [0,1] — penalize edges absent from prior network (-1 = model default)
#   force_basins   : float in [0,1] — preserve mode means in NB mixture (-1 = model default)
#   temporal_basins: 0 or 1 — enforce temporal mode consistency
#   rd             : 0/1 — run read-depth correction step (default 0)
#   ref            : 0/1 — run prepare_reference_network step (default 0)
#   test           : 0/1 — run infer_test + check_test_to_train (default 0)
#   kov            : 0/1 — run simulate_network_KOV + check_KOV (default 1)

input_dir="$1"
split="${2:-full}"
change="${3:-0}"
rate="${4:-0}"
mean="${5:--1}"
stimulus="${6:--1}"
prior="${7:--1}"
force_basins="${8:--1}"
temporal_basins="${9:--1}"
rd="${10:-0}"
ref="${11:-0}"
test="${12:-0}"
kov="${13:-1}"

if [ "$rd" = "1" ]; then
    echo "Inference rd"
    python infer_rd.py -i "${input_dir}"
fi

echo "Select DE genes and split cells"
python select_DEgenes_and_split.py -i "${input_dir}" -c "${change}" -r "${rate}" -s "${split}" -m "${mean}"

if [ "$ref" = "1" ]; then
    echo "Compute prior network"
    python prepare_reference_network.py -i "${input_dir}" -d 4
fi

echo "Get kinetic rates"
python get_kinetic_rates.py -i "${input_dir}" -s "${split}"

echo "Inference mixture"
python infer_mixture.py -i "${input_dir}" -s "${split}" -m "${mean}" -f "${force_basins}" -t "${temporal_basins}"

echo "Check mixture"
python check_mixture_to_data.py -i "${input_dir}" -s "${split}"

echo "Infer network structure"
python infer_network_structure.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}" -f "${force_basins}" -b "${temporal_basins}"

echo "Adapt network to simulate and degradation rates"
python infer_network_simul.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}"

echo "Simulate network"
python simulate_network.py -i "${input_dir}" -s "${split}"

echo "Check simulation"
python check_sim_to_data.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}"

if [ "$test" = "1" ]; then
    echo "Infer and simulate test"
    python infer_test.py -i "${input_dir}" --stimulus "${stimulus}" --prior "${prior}" -f "${force_basins}" -b "${temporal_basins}"
    python check_test_to_train.py -i "${input_dir}" -s "${split}"
fi

if [ "$kov" = "1" ]; then
    echo "Simulate KOV"
    python simulate_network_KOV.py -i "${input_dir}" -s "${split}"
    echo "Check KOV"
    python check_KOV_to_sim.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}"
fi

echo "All scripts executed !"
