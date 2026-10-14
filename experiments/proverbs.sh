#!/usr/bin/env bash
# «Нейро-Крокодил»: художник передаёт поговорку только изображением, слепой зритель угадывает.
#
# Художники различаются только генератором и языком промпта:
#   A — SDXL-Turbo, английский; B — Yandex ART, русский; C — Yandex ART, английский.
# Англоязычному художнику приходится переводить: дословно («hang noodles on ears»),
# через английский аналог с другой образностью («don't count your chickens») или через смысл.
# Вопросы: узнаёт ли зритель поговорку по картине и как перевод на это влияет.
#
# Запуск из корня проекта: bash experiments/proverbs.sh
set -euo pipefail

REPEATS=${REPEATS:-2}
ITERATIONS=${ITERATIONS:-2}
ROOT=${ROOT:-runs/proverbs_$(date +%Y%m%d_%H%M%S)}

PROVERBS=(
  "Делить шкуру неубитого медведя"     # англ. аналог с другими образами: count your chickens
  "Как слон в посудной лавке"          # почти тот же образ: bull in a china shop
  "Когда рак на горе свистнет"         # англ. аналог: when pigs fly
  "Вешать лапшу на уши"                # аналог без образа: pull someone's leg
  "Медвежья услуга"                    # прямого аналога нет
  "Рукописи не горят"                  # крылатая фраза из «Мастера и Маргариты»
)

index=0
for proverb in "${PROVERBS[@]}"; do
  index=$((index + 1))
  agent-artist-experiment run \
    --idea "$proverb" \
    --artists A,B,C \
    --modes isolated \
    --repeats "$REPEATS" \
    --iterations "$ITERATIONS" \
    --start-rounds 2 \
    --output-dir "$ROOT/$(printf '%02d' "$index")"
done

{
  agent-artist-experiment summary "$ROOT"
  agent-artist-experiment recover "$ROOT"
  agent-artist-experiment translation "$ROOT"
} | tee "$ROOT/summary.md"
