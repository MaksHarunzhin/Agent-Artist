#!/usr/bin/env bash
# «Опытный художник»: помогает ли знание типичных промахов генератора.
#
# Карта непосильных деталей показала: SDXL-Turbo теряет заказанные объекты, Yandex ART
# промахивается с расположением и светом. Опытный художник (--experienced) получает
# советы именно об этом. Сравниваем с обычным на тех же идеях: оценка эксперта,
# «угадай заказ» и доли категорий замечаний.
#
# Запуск из корня проекта: bash experiments/experienced.sh
set -euo pipefail

REPEATS=${REPEATS:-2}
ITERATIONS=${ITERATIONS:-2}
ROOT=${ROOT:-runs/experienced_$(date +%Y%m%d_%H%M%S)}

index=0
for idea in "Маяк в шторм" "Лиса в зимнем лесу"; do
  index=$((index + 1))
  for variant in plain experienced; do
    flag=$([ "$variant" = experienced ] && echo --experienced || true)
    agent-artist-experiment run $flag \
      --idea "$idea" --artists A,B --modes isolated \
      --repeats "$REPEATS" --iterations "$ITERATIONS" --start-rounds 2 \
      --no-describe \
      --output-dir "$ROOT/$(printf '%02d' "$index")_$variant"
  done
done

for variant in plain experienced; do
  echo "# $variant"
  agent-artist-experiment summary $ROOT/*_"$variant"
  agent-artist-experiment recover $ROOT/*_"$variant"
  agent-artist-experiment limits $ROOT/*_"$variant"
done | tee "$ROOT/summary.md"
