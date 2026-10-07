#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
session="cheese-real-sep2026"
if tmux has-session -t "=$session" 2>/dev/null; then
    echo "Session $session already exists; attach with: tmux attach -t $session"
    exit 1
fi
mkdir -p results
python_bin="$(command -v python)"
printf -v worker '%q -u scripts/search_cheese_real.py --quality accurate' "$python_bin"
printf -v monitor 'watch -n 10 %q scripts/search_cheese_real.py --status' "$python_bin"
tmux new-session -d -s "$session" -n search -c "$repo_dir"
tmux set-option -t "$session" remain-on-exit on
tmux send-keys -t "$session:search" "set -o pipefail; $worker 2>&1 | tee -a results/search.log; code=\${PIPESTATUS[0]}; printf '%s\\n' \"\$code\" > results/exit_code; exit \"\$code\"" Enter
tmux new-window -t "$session" -n progress -c "$repo_dir" "$monitor"
echo "Started: tmux attach -t $session"
echo "Status: $python_bin scripts/search_cheese_real.py --status"
