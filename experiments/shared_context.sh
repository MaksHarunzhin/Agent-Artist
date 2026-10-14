#!/usr/bin/env bash
# Главный вопрос проекта с запасом точности: помогает ли художникам общий контекст.
#
# Первая серия (runs/experiment_lighthouse) дала тенденцию в пользу shared при 2 прогонах на
# режим — ниже шумового пола. Здесь 5 повторов на режим и панель из трёх оценок эксперта
# (медиана), переписывание после отказа модерации включено по умолчанию.
#
# Запуск из корня проекта: bash experiments/shared_context.sh
set -euo pipefail

REPEATS=${REPEATS:-5}
ITERATIONS=${ITERATIONS:-3}
ROOT=${ROOT:-runs/shared_context_$(date +%Y%m%d_%H%M%S)}

agent-artist-experiment run \
  --idea "Маяк в шторм" --artists A,B --modes isolated,shared \
  --repeats "$REPEATS" --iterations "$ITERATIONS" --start-rounds 3 \
  --expert-votes 3 --no-describe \
  --output-dir "$ROOT"

{
  agent-artist-experiment summary "$ROOT"
  agent-artist-experiment recover "$ROOT"
} | tee "$ROOT/summary.md"
