#!/bin/sh
# Assemble the static site in _site: the page, its recorded episodes, and the README's videos
# and pipeline figure (kept in docs/, not duplicated in site/). Used by the Pages workflow too.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
out="$root/_site"
rm -rf "$out"
mkdir -p "$out/videos" "$out/figures"
cp -R "$root/site/." "$out/"
rm -f "$out/build.sh"
cp "$root"/docs/videos/fly-brain-highlights.mp4 "$root"/docs/videos/fly-brain-unseen-objects.mp4 \
  "$root"/docs/videos/fly-brain-unseen-composition.mp4 "$root"/docs/videos/kitchen-four-tasks-perturbed.mp4 \
  "$out/videos/"
cp "$root/docs/figures/pipeline-figure.png" "$out/figures/"
touch "$out/.nojekyll"
echo "$out"
