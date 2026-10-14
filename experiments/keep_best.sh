#!/usr/bin/env bash
# «Художник не забывает лучшее»: становится ли цикл критики обучением, а не пересэмплированием.
#
# Только художник A (локальный SDXL — генерация бесплатная), 4 итерации без ранней остановки.
# Обычный художник правит последнюю попытку, с --keep-best — лучшую. Сравниваем, насколько
# последняя картина лучше первой.
#
# Запуск из корня проекта: bash experiments/keep_best.sh
set -euo pipefail

REPEATS=${REPEATS:-3}
ITERATIONS=${ITERATIONS:-4}
ROOT=${ROOT:-runs/keep_best_$(date +%Y%m%d_%H%M%S)}

index=0
for idea in "Маяк в шторм" "Лиса в зимнем лесу"; do
  index=$((index + 1))
  for variant in plain keep_best; do
    flag=$([ "$variant" = keep_best ] && echo --keep-best || true)
    agent-artist-experiment run $flag \
      --idea "$idea" --artists A --modes isolated \
      --repeats "$REPEATS" --iterations "$ITERATIONS" --start-rounds 2 \
      --no-describe \
      --output-dir "$ROOT/$(printf '%02d' "$index")_$variant"
  done
done

for variant in plain keep_best; do
  echo "# $variant"
  agent-artist-experiment summary $ROOT/*_"$variant"
done | tee "$ROOT/summary.md"
