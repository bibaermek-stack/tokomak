#!/usr/bin/env bash
# Silent master + voice-over -> final reel (video stream copied untouched).
set -euo pipefail
cd "$(dirname "$0")"
[ -f vo_track.wav ] || bash vo.sh
ffmpeg -loglevel error -y -i out/sintez-reel-silent.mp4 -i vo_track.wav \
  -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -ar 44100 -ac 2 -shortest -movflags +faststart out/sintez-reel.mp4
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate:format=duration -of compact out/sintez-reel.mp4
