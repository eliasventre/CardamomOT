#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cardamom_light

# Usage: ./run.sh <input_dir> [stimulus] [prior] [mean_forcing] [force_basins] [temporal_basins]
#
# Only the hard-to-calibrate parameters are given here, in this order (empty or -1 = value of the
# Model_parameters sheet of <input_dir>/Data/CardamomOT_inputs.xlsx, else default of
# CardamomOT/model/base.py; precedence: default < workbook < command line):
#   stimulus        : penalisation of the stimulus edges, in [0, 1]           (model.stimulus)
#   prior           : weight of the edges absent from the literature prior    (model.prior_network_pen)
#                     (0 = hard constraint, sparse; 1 = prior ignored)
#   mean_forcing    : mean-forcing intensity of the NB mixture                 (model.mean_forcing_em)
#   force_basins    : preservation of the basin weights in the network fit    (model.force_basins)
#   temporal_basins : temporal consistency of the basins, 0 or 1              (model.temporal_basins)
#
# Everything else is a parameter of the Model_parameters sheet (or of base.py), e.g.
#   split ('train' / 'full'), train_rate, select_genes, build_prior_network, estimate_proliferation_rates,
#   run_test, simulate_perturbations, simulate_with_proliferation, species.
# Same as: cardamomot pipeline -i <input_dir> [--stimulus ..] [--prior ..] [--mean-forcing ..] ...

# --report-only (anywhere after <input_dir>): rebuild the final report with the current visualisation parameters of
# the workbook (embedding_method_visualization, cell_depth_for_representation, classifier_method, report_*): every check
# script of the run, the classical OT if from an earlier version, then the report (no inference step)
# --from <step> (anywhere after <input_dir>): resume the pipeline at <step>, e.g. --from infer_network_simul
input_dir="$1"
if [ -z "$input_dir" ]; then
    echo "Usage: ./run.sh <input_dir> [stimulus] [prior] [mean_forcing] [force_basins] [temporal_basins] [--report-only] [--from <step>]"
    exit 1
fi
shift
args=()
extra=()
while [ $# -gt 0 ]; do
    case "$1" in
        --report-only) extra+=("--report-only") ;;
        --from) extra+=("--from" "$2"); shift ;;
        *) args+=("$1") ;;
    esac
    shift
done
set -- "$input_dir" "${args[@]}"
opts=("${extra[@]}")
names=(stimulus prior mean-forcing force-basins temporal-basins)
values=("$2" "$3" "$4" "$5" "$6")
for k in 0 1 2 3 4; do
    if [ -n "${values[$k]}" ]; then
        opts+=("--${names[$k]}" "${values[$k]}")
    fi
done

python -m CardamomOT.cli pipeline -i "${input_dir}" "${opts[@]}"
