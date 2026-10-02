#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cardamom_light

# Usage: ./run.sh <input_dir> <split> <rate> <change> <mean> [stimulus]
#                 [force_basins] [temporal_basins] [ref] [prior] [test] [kov] [simulate_proliferation]
#                 [use_proliferation] [--species human|mouse]
#
#   split                 : full | train
#   rate                  : float — share of the cells in the train split
#   change                : 0/1 — gene selection (1: OTVelo network + Steiner tree to Data/genes_queries.txt, budget model.num_max_genes;
#                            also writes the literature prior cardamomOT/ref_network.csv)
#   mean_forcing          : float — mean-forcing intensity for NB mixture (model default 0.5, -1 = use model default)
#   stimulus              : float in [0,1] — penalize stimulus edges (-1 = model default)
#   force_basins          : float in [0,1] — preserve mode means in NB mixture (-1 = model default)
#   temporal_basins       : 0 or 1 — enforce temporal mode consistency
#   ref                   : 0/1 — build the literature prior (default 0). With change=1 the selection builds it;
#                            with change=1 and prior=0 the gene budget is set by model.max_free_params
#   prior                 : float in [0,1] — weight of the edges absent from the prior network
#                            (0 = hard constraint, sparse; 1 = prior ignored; -1 = model default)
#   test                  : 0/1 — run infer_test + check_test_to_train (default 0)
#   kov                   : 0/1 — run simulate_network_KOV + check_KOV (default 1)
#   simulate_proliferation : 0/1 — simulate with proliferation/death (branching; trains the R(P) MLP
#                            in infer_network_simul, used only by the simulations) (default 0)
#   use_proliferation     : 0/1 — run get_proliferation_rates to (re)estimate
#                            obs['proliferation_net_rate'] from literature gene
#                            signatures (default 1)
#   --species             : organism, human or mouse — trailing flag, e.g.
#                            ./run.sh my_project full 0.7 0 0.5 --species mouse.
#                            If omitted, get_proliferation_rates and get_degradation_rates
#                            detect it from gene names (Mki67/Gata1 = mouse, MKI67/GATA1 = human)

# --species is a named flag and can appear anywhere in the argument list;
# pull it out first so the remaining positional arguments line up as before.
species=""
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
force_basins="${7:--1}"
temporal_basins="${8:--1}"
ref="${9:-0}"
prior="${10:--1}"
test="${11:-0}"
kov="${12:-1}"
simulate_proliferation="${13:-0}"
use_proliferation="${14:-0}"

# Build --simulate-proliferation flag string used by infer_network_simul, simulate_network, simulate_network_KOV
prolif_flag=""
if [ "$simulate_proliferation" = "1" ]; then
    prolif_flag="--simulate-proliferation"
fi

echo "Estimate cell depth"
python estimate_cell_depth.py -i "${input_dir}"

if [ "$use_proliferation" = "1" ]; then
    echo "Get proliferation rates"
    python get_proliferation_rates.py -i "${input_dir}" ${species:+--species "${species}"}
fi

echo "Select genes and split cells"
python select_genes_and_split.py -i "${input_dir}" -s "${split}" -r "${rate}" -c "${change}" --mean-forcing "${mean_forcing}" --prior "${prior}" --ref "${ref}"

# With change=1 the selection already wrote the literature prior (same computation)
if [ "$ref" = "1" ] && [ "$change" != "1" ]; then
    echo "Build prior network"
    python build_reference_network.py -i "${input_dir}" ${species:+--species "${species}"}
fi

echo "Get degradation rates"
python get_degradation_rates.py -i "${input_dir}" -s "${split}" ${species:+--species "${species}"}

echo "Inference mixture"
python infer_mixture.py -i "${input_dir}" -s "${split}" --mean-forcing "${mean_forcing}" --force-basins "${force_basins}" --temporal-basins "${temporal_basins}"

echo "Check mixture"
python check_mixture_to_data.py -i "${input_dir}" -s "${split}"

echo "Infer network structure"
python infer_network_structure.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}" --force-basins "${force_basins}" --temporal-basins "${temporal_basins}"
# Exit code 3 = stationary data (no/single timepoint), handled by CardamomOT-stat
if [ $? -eq 3 ]; then
    echo "Stopping pipeline: switch to method CardamomOT-stat, in prep."
    exit 0
fi

echo "Adapt network to simulate and degradation rates"
python infer_network_simul.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}" $prolif_flag

echo "Simulate network"
python simulate_network.py -i "${input_dir}" -s "${split}" $prolif_flag

echo "Check simulation"
python check_sim_to_data.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}"

if [ "$test" = "1" ]; then
    echo "Infer and simulate test"
    python infer_test.py -i "${input_dir}" --stimulus "${stimulus}" --prior "${prior}" --force-basins "${force_basins}" --temporal-basins "${temporal_basins}"
    python check_test_to_train.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}"
fi

if [ "$kov" = "1" ]; then
    echo "Simulate KOV"
    python simulate_network_KOV.py -i "${input_dir}" -s "${split}" $prolif_flag
    echo "Check KOV"
    python check_KOV_to_sim.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}"
fi

echo "Final report"
python report_results.py -i "${input_dir}" -s "${split}" --stimulus "${stimulus}" --prior "${prior}"

echo "All scripts executed !"
