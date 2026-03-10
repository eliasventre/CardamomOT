#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cardamom_light

# prefer using the package entry point where possible
# PYTHON=/usr/bin/python3

input_dir="$1"
split="$2"
change="$3"
rate="$4"
mean="$5"

# echo "Inference rd"
# python infer_rd.py -i "${input_dir}" 

# echo "Select DE genes and split cells"
# python select_DEgenes_and_split.py -i "${input_dir}" -c "${change}" -r "${rate}" -s "${split}" -m "${mean}"

# echo "Compute prior network"
# python prepare_reference_network.py -i "${input_dir}" -d 3

# echo "Get kinetic rates"
# python get_kinetic_rates.py -i "${input_dir}" -s "${split}"

# echo "Inference mixture"
# python infer_mixture.py -i "${input_dir}" -s "${split}" -m "${mean}"

# echo "Check mixture"
# python check_mixture_to_data.py -i "${input_dir}" -s "${split}" 

# echo "Infer network structure"
# python infer_network_structure.py -i "${input_dir}" -s "${split}" 

# echo "Adapt network to simulate and degradation rates"
# python infer_network_simul.py -i "${input_dir}" -s "${split}" 

# echo "Simulate network"
# python simulate_network.py -i "${input_dir}" -s "${split}"

echo "Check simulation"
python check_sim_to_data.py -i "${input_dir}" -s "${split}" 

# if [ "$split" != "full" ]; then
#     echo "Infer and simulate test"
#     python infer_test.py -i "${input_dir}"
#     python check_test_to_train.py -i "${input_dir}" -s "${split}" 
# fi

echo "Simulate KOV"
python simulate_network_KOV.py -i "${input_dir}" -s "${split}"

echo "Check KOV"
python check_KOV_to_sim.py -i "${input_dir}" -s "${split}" 

echo "All scripts executed !"