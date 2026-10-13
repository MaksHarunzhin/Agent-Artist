#!/usr/bin/env bash
# Культурная асимметрия: теряется ли культурный контекст в английском промпте?
#
# Художники различаются только генератором и языком промпта:
#   A — SDXL-Turbo, английский
#   B — Yandex ART, русский
#   C — Yandex ART, английский
# B против C — эффект языка при одном генераторе, A против C — эффект генератора.
# Гипотеза: на культурно-специфичных идеях разрыв B − C больше, чем на нейтральных —
# и в оценке эксперта, и в том, узнаёт ли слепой зритель идею по картине.
#
# Запуск из корня проекта: bash experiments/culture_language.sh
# Стоимость: 6 идей × REPEATS × ITERATIONS × 2 картинки Yandex ART (B и C).
set -euo pipefail

REPEATS=${REPEATS:-2}
ITERATIONS=${ITERATIONS:-2}
ROOT=${ROOT:-runs/culture_$(date +%Y%m%d_%H%M%S)}

CULTURAL_IDEAS=(
  "Масленица на деревенской площади"
  "Баба-Яга у избушки на курьих ножках"
  "Двор хрущёвки зимним вечером"
  "Мороз и солнце; день чудесный"
)
NEUTRAL_IDEAS=(
  "Маяк в шторм"
  "Лиса в зимнем лесу"
)

index=0
for idea in "${CULTURAL_IDEAS[@]}" "${NEUTRAL_IDEAS[@]}"; do
  index=$((index + 1))
  agent-artist-experiment run \
    --idea "$idea" \
    --artists A,B,C \
    --modes isolated \
    --repeats "$REPEATS" \
    --iterations "$ITERATIONS" \
    --start-rounds 2 \
    --output-dir "$ROOT/$(printf '%02d' "$index")"
done

# Оценки эксперта и «угадай заказ»: узнаёт ли слепой зритель культурную тему по картине
{
  agent-artist-experiment summary "$ROOT"
  agent-artist-experiment recover "$ROOT"
} | tee "$ROOT/summary.md"
