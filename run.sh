#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cardamom_light

# prefer using the package entry point where possible
# PYTHON=/usr/bin/python3

input_dir="$1"
split="${2:-full}"
change="${3:-0}"
rate="${4:-0}"
mean="${5:--1}"
stimulus="${6:--1}"
prior="${7:--1}"
force_basins="${8:--1}"
temporal_basins="${9:--1}"

# echo "Inference rd"
# python infer_rd.py -i "${input_dir}"

# echo "Select DE genes and split cells"
# python select_DEgenes_and_split.py -i "${input_dir}" -c "${change}" -r "${rate}" -s "${split}" -m "${mean}"

# echo "Compute prior network"
# python prepare_reference_network.py -i "${input_dir}" -d 4

# echo "Get kinetic rates"
# python get_kinetic_rates.py -i "${input_dir}" -s "${split}"

# echo "Inference mixture"
# python infer_mixture.py -i "${input_dir}" -s "${split}" -m "${mean}" -f "${force_basins}" -t "${temporal_basins}"

# echo "Check mixture"
# python check_mixture_to_data.py -i "${input_dir}" -s "${split}"

# echo "Infer network structure"
# python infer_network_structure.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}" -f "${force_basins}" -b "${temporal_basins}"

echo "Adapt network to simulate and degradation rates"
python infer_network_simul.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}"

echo "Simulate network"
python simulate_network.py -i "${input_dir}" -s "${split}"

echo "Check simulation"
python check_sim_to_data.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}"

# if [ "$split" != "full" ]; then
#     echo "Infer and simulate test"
#     python infer_test.py -i "${input_dir}" --stimulus "${stimulus}" --prior "${prior}" -f "${force_basins}" -b "${temporal_basins}"
#     python check_test_to_train.py -i "${input_dir}" -s "${split}"
# fi

echo "Simulate KOV"
python simulate_network_KOV.py -i "${input_dir}" -s "${split}"

echo "Check KOV"
python check_KOV_to_sim.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}"

echo "All scripts executed !"