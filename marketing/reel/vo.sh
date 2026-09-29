#!/usr/bin/env bash
# Kazakh voice-over for the reel: Piper TTS with the ISSAI KazakhTTS voice
# (speaker M1 "Iseke", a native male voice), each line fitted to its window
# and placed on the 18.3 s timeline, mastered to -14 LUFS for Reels.
# Needs internet the first time (model from huggingface.co). Output: vo_track.wav
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p work && cd work
pip install -q piper-tts >/dev/null 2>&1 || true
B=https://huggingface.co/rhasspy/piper-voices/resolve/main/kk/kk_KZ/issai/high
[ -f kk.onnx ] || { curl -sfL -o kk.onnx.json $B/kk_KZ-issai-high.onnx.json; curl -sfL -o kk.onnx $B/kk_KZ-issai-high.onnx; }
cp ../vo_plan.json plan.json
python3 - <<'PY'
import json, wave
from piper import PiperVoice
try:
    from piper import SynthesisConfig
except Exception:
    SynthesisConfig = None
v = PiperVoice.load('kk.onnx'); plan = json.load(open('plan.json'))
for i, L in enumerate(plan):
    with wave.open(f'L{i}.wav', 'wb') as wf:
        if SynthesisConfig:
            v.synthesize_wav(L['text'], wf, syn_config=SynthesisConfig(
                speaker_id=1, length_scale=L['ls'], noise_scale=0.6, noise_w_scale=0.7))
        else:
            v.synthesize(L['text'], wf, speaker_id=1, length_scale=L['ls'], noise_scale=0.6, noise_w=0.7)
PY
n=$(python3 -c "import json;print(len(json.load(open('plan.json'))))")
args=""; fl=""; mix=""
for ((i=0;i<n;i++)); do
  t=$(python3 -c "import json;print(int(json.load(open('plan.json'))[$i]['t']*1000))")
  ffmpeg -loglevel error -y -i L$i.wav -af "silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse" -ar 44100 T$i.wav
  args="$args -i T$i.wav"; fl="$fl[$i:a]adelay=${t}|${t}[a$i];"; mix="$mix[a$i]"
done
ffmpeg -loglevel error -y $args -filter_complex \
  "${fl}${mix}amix=inputs=$n:normalize=0,apad=whole_dur=18.3,atrim=0:18.3,highpass=f=70,acompressor=threshold=-20dB:ratio=3:attack=5:release=120,loudnorm=I=-14:TP=-1.0:LRA=7[out]" \
  -map "[out]" -ac 1 -ar 44100 ../vo_track.wav
echo "vo_track.wav: $(ffprobe -v error -show_entries format=duration -of csv=p=0 ../vo_track.wav) s"
