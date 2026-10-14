#!/usr/bin/env bash
# «Испорченный телефон»: насколько долго смысл переживает пересказы через картинку.
#
# Для каждой идеи — цепочка у каждого художника (A — SDXL, англ.; B — Yandex ART, рус.;
# C — Yandex ART, англ.). Нейтральная идея, культурная и поговорка: ожидаем, что
# поговорка теряет смысл первой (рисуют буквальные образы), а нейтральная держится дольше.
#
# Запуск из корня проекта: bash experiments/telephone.sh
set -euo pipefail

STEPS=${STEPS:-6}
ROOT=${ROOT:-runs/telephone_$(date +%Y%m%d_%H%M%S)}

IDEAS=(
  "Маяк в шторм"
  "Баба-Яга у избушки на курьих ножках"
  "Делить шкуру неубитого медведя"
)

index=0
for idea in "${IDEAS[@]}"; do
  index=$((index + 1))
  agent-artist-experiment telephone \
    --idea "$idea" \
    --artists A,B,C \
    --steps "$STEPS" \
    --output-dir "$ROOT/$(printf '%02d' "$index")"
done
