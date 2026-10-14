# Агент-художник

Художники-агенты уточняют детали картины у арт-эксперта (заказчика), пишут промпты для
генераторов изображений и переделывают работу по его критике. Главный вопрос
эксперимента — **асимметрия контекста**: помогает ли художникам видеть диалог эксперта
с другими художниками, и сколько из заказа вообще доходит до картины.

Результаты экспериментов — в [`docs/results.md`](docs/results.md), что и зачем добавлено —
в [`docs/features.md`](docs/features.md), разбор данных — в ноутбуке
[`Agent_Artist.ipynb`](Agent_Artist.ipynb).

**Материалы для защиты** (приватные ссылки, доступ выдаёт владелец через Share):
[презентация](https://claude.ai/artifact/Pcir1t4DRNuMDQYxFDSjLa) ·
[слепое голосование «Две картины, одна идея»](https://claude.ai/artifact/NL9Jn4yeAKQCnpa6X5yAHK).

## Участники

| Роль | Модель | Где |
|---|---|---|
| Художник A | LLM `deepseek-v4.1-flash` + **SDXL-Turbo**, промпт на английском | LLM в Яндексе, генерация локально |
| Художник B | та же LLM + **Yandex ART**, промпт на русском | Яндекс |
| Художник C | та же LLM + **Yandex ART**, промпт на английском | Яндекс |
| Эксперт | `deepseek-v4.1-flash` отвечает на вопросы, `qwen3.6-35b-a3b` оценивает картину по изображению | Яндекс |
| Описание картин | VLM `qwen3-vl:8b` | локально, Ollama |

## Установка (macOS / Linux)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .

# Локальная VLM
brew install ollama          # или https://ollama.com
ollama serve                 # в отдельном терминале
ollama pull qwen3-vl:8b
```

SDXL-Turbo (~7 ГБ) скачается при первом запуске. На Mac с Apple Silicon генерация идёт на
GPU (MPS), с NVIDIA — на CUDA, иначе на CPU.

Ключи Yandex AI Studio — в файле `.env` в корне проекта:

```
YANDEX_API_KEY=...
YANDEX_FOLDER_ID=...
```

Нужен сервисный аккаунт с доступом к генерации текста и изображений
(`ai.languageModels.user`, `ai.imageGeneration.user`).

## Запуск

Один прогон:

```bash
agent-artist --idea "Маяк в шторм" --max-iter 3              # художники A и B
agent-artist --idea "Масленица" --artists A,B,C --shared     # общий контекст
agent-artist --idea "Маяк в шторм" --expert child           # другой заказчик: minimalist, baroque, child, peredvizhnik
agent-artist --idea "Масленица" --no-describe --experienced   # без локальной VLM; художник знает слабости генератора
agent-artist --help                                           # все параметры
```

Результаты — в `runs/<дата_время>/`: `log.json` (весь диалог, промпты, оценки),
картины, `timelapse.webp` / `.gif` и `evolution.png`.

Эксперименты и анализ:

```bash
agent-artist-experiment run --idea "Маяк в шторм" --repeats 3 --iterations 3   # shared против isolated
agent-artist-experiment telephone --idea "Делить шкуру неубитого медведя" --steps 6

agent-artist-experiment summary runs/…       # оценки эксперта: таблицы по режимам и итерациям
agent-artist-experiment recover runs/…       # «угадай заказ»: что из заказа дошло до картины
agent-artist-experiment translation runs/…   # как английский промпт перенёс русскую идею
agent-artist-experiment limits runs/…        # карта непосильных деталей: в чём промахиваются генераторы
agent-artist-experiment gallery runs/… --label expert --output gallery.png   # лучшие картины сеткой
agent-artist-experiment drift runs/…         # движущаяся мишень: первая и последняя картина под одним контекстом
agent-artist-experiment noise runs/…         # шумовой пол: разброс оценок той же картины и того же промпта
agent-artist-experiment timelapse runs/…     # таймлапсы для готовых прогонов
```

Готовые серии — в `experiments/`: `culture_language.sh`, `proverbs.sh`, `telephone.sh`,
`clients.sh`, `experienced.sh`, `keep_best.sh`. Анализ с графиками — ноутбук `Agent_Artist.ipynb`
(`pip install -e ".[analysis]"`).

## Тесты

```bash
pip install -e ".[dev]"
pytest
```

Тесты офлайн, без вызовов моделей: разбор ответов LLM, повторы при сбоях, переписывание
после отказа модерации, подгонка промпта под лимит, контекст художника, сводки, настройки.

## Как устроен прогон

1. **Согласование.** Художники по очереди задают эксперту вопросы; эксперт отвечает,
   исходя из своих вкусов. В режиме `--shared` художники видят и чужие вопросы.
2. **Промпт.** Художник пишет промпт под ограничения своего генератора (77 токенов CLIP у
   SDXL, 500 символов у Yandex ART) и сокращает его сам, если не уложился.
3. **Картина** — локально или в Yandex ART.
4. **Оценка.** Эксперт смотрит на картину и ставит 1–10 с конкретными замечаниями.
5. После критики — новый раунд вопросов и следующая итерация, пока все не наберут
   целевую оценку или не кончатся итерации.

## Код

```
src/agent_artist/
  cli.py           agent-artist: аргументы → Settings → ArtStudio
  config.py        Settings, художники A/B/C, вкусы эксперта
  studio.py        ArtStudio — основной цикл; write_fitting_prompt
  dialogue.py      кто какие вопросы и ответы видит
  prompts.py       все системные промпты
  llm.py           YandexLLM: JSON-ответы, повторы при сбоях, картинки на вход
  vision.py        локальная VLM в Ollama
  generators/      SDXL-Turbo и Yandex ART с проверкой лимитов промпта
  experiment.py    agent-artist-experiment: серии и сводки
  recovery.py      «угадай заказ»
  translation.py   стратегии перевода идеи
  limits.py        карта непосильных деталей
  drift.py         движущаяся мишень
  noise.py         шумовой пол
  summary.py       сводка оценок эксперта
  analysis.py      прогоны → таблицы pandas для ноутбука
  runlogs.py       чтение логов, кеши, markdown-таблицы
  telephone.py     «испорченный телефон»
  timelapse.py     таймлапсы, сетки эволюции, галерея
```
