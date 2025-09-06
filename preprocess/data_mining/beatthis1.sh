#!/bin/bash

# CUDA 7번 고정
export CUDA_VISIBLE_DEVICES=7

echo "🔁 Running beat_this on multiple folders... (CUDA: $CUDA_VISIBLE_DEVICES)"

# 명령어 리스트
commands=(
  "beat_this duet_svs/24k/k_multitimbre/*.wav -o duet_svs/24k/k_multitimbre"
  "beat_this duet_svs/24k/NUS/*.wav -o duet_svs/24k/NUS"
  "beat_this duet_svs/24k/musdb_a_train/*.wav -o duet_svs/24k/musdb_a_train"
  "beat_this duet_svs/24k/OpenSinger/*.wav -o duet_svs/24k/OpenSinger"
  "beat_this duet_svs/24k/VocalSet/*.wav -o duet_svs/24k/VocalSet"
)

beat_this duet_svs/24k/OpenSinger/ -o duet_svs/24k/OpenSinger/ --touch-first --skip-existing
beat_this duet_svs/24k/OpenSinger/ -o duet_svs/24k/OpenSinger
# 하나씩 실행 (에러가 나도 다음으로 넘어감)
for cmd in "${commands[@]}"; do
  echo "▶️ $cmd"
  eval "$cmd" || echo "❌ Error occurred in: $cmd → skipping"
done

echo "✅ All commands finished (with or without errors)."
