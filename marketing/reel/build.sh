#!/usr/bin/env bash
# Capture the site, compose the reel, encode the silent master.
# Needs: node, a Chromium (CHROME=...), ffmpeg, and the site served on :8099.
set -euo pipefail
cd "$(dirname "$0")"
npm install --silent
( cd ../.. && python3 -m http.server 8099 >/dev/null 2>&1 & )
( python3 -m http.server 8098 >/dev/null 2>&1 & ); sleep 1
for clip in landing flight interior; do QUALITY=high FOVK_OUT=1.8 node capture.js http://localhost:8099/index.html ./cap --only=$clip & done
QUALITY=medium node capture.js http://localhost:8099/index.html ./cap --only=hud & wait
node render.js http://localhost:8098/composer.html ./frames
mkdir -p out
ffmpeg -loglevel error -y -framerate 30 -i frames/f_%04d.jpg -c:v libx264 -preset slow -crf 19 \
  -pix_fmt yuv420p -profile:v high -movflags +faststart out/sintez-reel-silent.mp4
echo "out/sintez-reel-silent.mp4"
