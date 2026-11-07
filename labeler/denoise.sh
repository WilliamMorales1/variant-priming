#!/usr/bin/env bash
# Writes <dir>_denoised/ next to each corpus dir given, leaving the originals untouched.
set -euo pipefail
for d in "${@:-corpus/model corpus/model_alt}"; do
  d=${d%/}
  mkdir -p "${d}_denoised"
  cp "$d"/*.lab "$d"/*.TextGrid "${d}_denoised/"
  for f in "$d"/*.wav; do
    ffmpeg -y -loglevel error -i "$f" -af "highpass=f=60,afftdn=nr=10:nf=-55:nt=w:tn=1:tr=1" \
      -ar 16000 -ac 1 -c:a pcm_s16le "${d}_denoised/$(basename "$f")"
  done
done
