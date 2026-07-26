#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cardamom_light

# Usage: ./run.sh <input_dir> <split> <change> <rate> <mean> [stimulus] [prior]
#                 [force_basins] [temporal_basins] [ref] [test] [kov] [compute_proliferation]
#                 [use_proliferation] [--species human|mouse]
#
#   split                 : full | train
#   change                : 0/1 — differential gene selection
#   rate                  : float — rate parameter for kinetics
#   mean_forcing          : float — mean-forcing intensity for NB mixture (model default 0.5, -1 = use model default)
#   stimulus              : float in [0,1] — penalize stimulus edges (-1 = model default)
#   prior                 : float in [0,1] — penalize edges absent from prior network (-1 = model default)
#   force_basins          : float in [0,1] — preserve mode means in NB mixture (-1 = model default)
#   temporal_basins       : 0 or 1 — enforce temporal mode consistency
#   ref                   : 0/1 — run prepare_reference_network step (default 0)
#   test                  : 0/1 — run infer_test + check_test_to_train (default 0)
#   kov                   : 0/1 — run simulate_network_KOV + check_KOV (default 1)
#   compute_proliferation : 0/1 — learn R_opt MLP and simulate with branching (default 0)
#   use_proliferation     : 0/1 — run get_proliferation_rates to (re)estimate
#                            obs['proliferation_net_rate'] from literature gene
#                            signatures (default 1)
#   --species             : organism for get_proliferation_rates literature proliferation/death
#                            gene signatures, human or mouse (default: human) — trailing
#                            flag, e.g. ./run.sh my_project full 0.7 0 0.5 --species mouse

# --species is a named flag and can appear anywhere in the argument list;
# pull it out first so the remaining positional arguments line up as before.
species="human"
positional=()
while [ $# -gt 0 ]; do
    case "$1" in
        --species) species="$2"; shift 2 ;;
        *) positional+=("$1"); shift ;;
    esac
done
set -- "${positional[@]}"

input_dir="$1"
split="${2:-full}"
rate="${3:-1}"
change="${4:-0}"
mean_forcing="${5:--1}"
stimulus="${6:--1}"
prior="${7:--1}"
force_basins="${8:--1}"
temporal_basins="${9:--1}"
ref="${10:-0}"
test="${11:-0}"
kov="${12:-1}"
compute_proliferation="${13:-0}"
use_proliferation="${14:-0}"

Build --compute-proliferation flag string used by infer_network_simul, simulate_network, simulate_network_KOV
prolif_flag=""
if [ "$compute_proliferation" = "1" ]; then
    prolif_flag="--compute-proliferation"
fi

if [ "$use_proliferation" = "1" ]; then
    echo "Get proliferation rates"
    python get_proliferation_rates.py -i "${input_dir}" --species "${species}"
fi

echo "Select DE genes and split cells"
python select_DEgenes_and_split.py -i "${input_dir}" -s "${split}" -r "${rate}" -c "${change}" --mean-forcing "${mean_forcing}"


if [ "$ref" = "1" ]; then
    echo "Compute prior network"
    python prepare_reference_network.py -i "${input_dir}" -d 4
fi

echo "Get degradation rates"
python get_degradation_rates.py -i "${input_dir}" -s "${split}"

echo "Inference mixture"
python infer_mixture.py -i "${input_dir}" -s "${split}" --mean-forcing "${mean_forcing}" --force-basins "${force_basins}" --temporal-basins "${temporal_basins}"

echo "Check mixture"
python check_mixture_to_data.py -i "${input_dir}" -s "${split}"

echo "Infer network structure"
python infer_network_structure.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}" --force-basins "${force_basins}" --temporal-basins "${temporal_basins}"

echo "Adapt network to simulate and degradation rates"
python infer_network_simul.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}" $prolif_flag

echo "Simulate network"
python simulate_network.py -i "${input_dir}" -s "${split}" $prolif_flag

echo "Check simulation"
python check_sim_to_data.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}"

if [ "$test" = "1" ]; then
    echo "Infer and simulate test"
    python infer_test.py -i "${input_dir}" --stimulus "${stimulus}" --prior "${prior}" --force-basins "${force_basins}" --temporal-basins "${temporal_basins}"
    python check_test_to_train.py -i "${input_dir}" -s "${split}"
fi

if [ "$kov" = "1" ]; then
    echo "Simulate KOV"
    python simulate_network_KOV.py -i "${input_dir}" -s "${split}" $prolif_flag
    echo "Check KOV"
    python check_KOV_to_sim.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}"
fi

echo "All scripts executed !"
