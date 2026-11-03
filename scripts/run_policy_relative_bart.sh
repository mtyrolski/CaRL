#!/usr/bin/env bash
# Three matched scratch runs with persistent logs and a readable report.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run_dir="${1:-analysis/policy_relative_runs/scratch_$(TZ=Europe/Warsaw date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir"
finish() {
    local run_exit_code=$?
    local run_state=completed
    if (( run_exit_code != 0 )); then run_state=failed; fi
    printf '{"state":"%s","exit_code":%d}\n' "$run_state" "$run_exit_code" > "$run_dir/status.json"
}
trap finish EXIT
printf '{"state":"running","exit_code":null}\n' > "$run_dir/status.json"
cp configs/generator/sokoban_bart_small.json "$run_dir/model_config.json"
cp analysis/policy_relative_smoke_20261029/main/summary.json "$run_dir/calibration_metadata.json"
command=(env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. HF_HUB_OFFLINE=1
    uv run --frozen python -u scripts/train_policy_relative_bart.py
    --initialization scratch --model-config "$run_dir/model_config.json"
    --dataset analysis/policy_relative_smoke_20261029/main
    --conditions relative absolute none
    --steps 300 --batch 8 --lr 3e-4 --eval-every 100 --output "$run_dir")
printf '#!/usr/bin/env bash\n' > "$run_dir/command.sh"
printf '%q ' "${command[@]}" >> "$run_dir/command.sh"
printf '\n' >> "$run_dir/command.sh"
"${command[@]}" 2>&1 | tee "$run_dir/console.log"
uv run --frozen python scripts/summarise_policy_relative_bart.py "$run_dir" 2>&1 | tee "$run_dir/report.log"
printf 'Run and reports saved to %s\n' "$run_dir"
