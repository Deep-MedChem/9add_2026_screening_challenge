#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
session="cheese-enamine-oct2026"
run="results/cheese_enamine_all_r0_top100_oct2026_run"
input="data/enamine_all_r0_top100_oct2026.smi"
package="results/cheese_enamine_all_r0_top100_oct2026"
python_bin="$(command -v python)"

if [[ "${1:-}" == "--worker" ]]; then
    mkdir -p "$run"
    trap 'code=$?; printf "%s\n" "$code" > "$run/exit_code"' EXIT
    {
        "$python_bin" -u scripts/search_cheese_real.py --input "$input" --out "$run" --quality accurate
        "$python_bin" -u scripts/export_cheese_search.py --run "$run" --input "$input" --package "$package"
    } 2>&1 | tee -a "$run/search.log"
    exit 0
fi

if tmux has-session -t "=$session" 2>/dev/null; then
    echo "Session $session already exists; attach with: tmux attach -t $session"
    exit 1
fi
mkdir -p "$run"
printf -v worker 'bash %q --worker' "$repo_dir/scripts/start_cheese_enamine_tmux.sh"
printf -v monitor 'watch -n 10 %q scripts/search_cheese_real.py --out %q --status' "$python_bin" "$run"
tmux new-session -d -s "$session" -n search -c "$repo_dir"
tmux set-option -t "$session" remain-on-exit on
tmux send-keys -t "$session:search" "$worker; exit" Enter
tmux new-window -t "$session" -n progress -c "$repo_dir" "$monitor"
echo "Started: tmux attach -t $session"
echo "Log: $run/search.log"
echo "Automatic export on success: $package.zip"
