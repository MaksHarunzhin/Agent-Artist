#!/usr/bin/env bash
# «Разные заказчики»: одна идея — пять экспертов с разными вкусами.
#
# Если диалог с экспертом работает, картины у кинематографиста, минималиста, любителя
# барокко, шестилетнего ребёнка и передвижника должны заметно различаться.
# Результат — галерея clients.png: художники × заказчики, лучшая картина каждого.
#
# Запуск из корня проекта: bash experiments/clients.sh
set -euo pipefail

IDEA=${IDEA:-"Маяк в шторм"}
ITERATIONS=${ITERATIONS:-2}
ROOT=${ROOT:-runs/clients_$(date +%Y%m%d_%H%M%S)}

for expert in cinematic minimalist baroque child peredvizhnik; do
  agent-artist \
    --idea "$IDEA" \
    --expert "$expert" \
    --artists A,B \
    --no-describe \
    --start-rounds 2 \
    --min-iter "$ITERATIONS" --max-iter "$ITERATIONS" \
    --output-dir "$ROOT/$expert"
done

agent-artist-experiment gallery "$ROOT" --label expert \
  --title "«$IDEA» у пяти заказчиков" --output "$ROOT/clients.png"
