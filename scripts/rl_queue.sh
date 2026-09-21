#!/usr/bin/env bash
# Run the harder-task PPO experiments one at a time after the first PPO run finishes.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
until grep -q '"status": "\(complete\|failed\)"' runs/ppo-pick-place-001/results.json 2>/dev/null; do
  sleep 60
done
for name in randomized randomized-shuffled; do
  output=runs/ppo-pick-place-$name-001
  [ -e "$output" ] && continue
  echo "[$(date '+%F %T')] start $output"
  uv run --no-sync flyarm rl ppo --config configs/ppo-pick-place-$name.json --output "$output" \
    > "$output.log" 2>&1
  echo "[$(date '+%F %T')] exit $? $output"
done
