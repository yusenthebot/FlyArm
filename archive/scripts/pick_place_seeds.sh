#!/usr/bin/env bash
# Pick-and-place topology replication: seeds 3-5 with two shuffles each, then a second
# independent shuffle for seeds 0-2 (their first shuffle is in runs 001 and 002).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
FLYARM="uv run --no-sync flyarm"
for pair in "seeds345:whole-brain-pick-place-003" "shuffle-r1:whole-brain-pick-place-004"; do
  config=configs/whole-brain-pick-place-${pair%%:*}.json
  output=runs/${pair##*:}
  [ -e "$output" ] && continue
  echo "[$(date '+%F %T')] start $output"
  $FLYARM whole-brain run --config "$config" --output "$output" > "$output.log" 2>&1
  echo "[$(date '+%F %T')] exit $? $output"
done
