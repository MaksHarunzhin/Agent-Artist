# Агент-художник

Мультиагентная система: художники-агенты уточняют заказ у арт-эксперта, пишут промпты для
генераторов изображений и переделывают картину по его критике. Главный вопрос —
**асимметрия контекста**: помогает ли художнику видеть чужой диалог с экспертом и сколько из
заказа вообще доходит до картины.

**Сайт с выставкой экспериментов:** https://maksharunzhin.github.io/Agent-Artist/

| Художник | Генератор | Язык промпта |
|---|---|---|
| A | SDXL-Turbo, локально | английский |
| B | Yandex ART | русский |
| C | Yandex ART | английский |

Эксперт — LLM в Yandex AI Studio, оценивает саму картину мультимодальной моделью.

## Быстрый старт

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis,dev]"
echo "YANDEX_API_KEY=...
YANDEX_FOLDER_ID=..." > .env

agent-artist --idea "Маяк в шторм" --max-iter 3 --no-describe   # один прогон, результат в runs/
agent-artist-experiment --help                                   # серии и анализ
pytest                                                           # офлайн-тесты
```

SDXL-Turbo (~7 ГБ) скачается при первом запуске и работает на Apple Silicon (MPS), CUDA или CPU.

## Документы

- [`REPORT.md`](REPORT.md) — отчёт по работе
- [`docs/results.md`](docs/results.md) — результаты экспериментов с погрешностями
- [`docs/features.md`](docs/features.md) — что добавлено и почему
- [`docs/usage.md`](docs/usage.md) — все команды, флаги и устройство кода
- [`docs/prompts.md`](docs/prompts.md) — как мы работали с кодинг-агентом
- [`Agent_Artist.ipynb`](Agent_Artist.ipynb) — анализ данных с графиками
