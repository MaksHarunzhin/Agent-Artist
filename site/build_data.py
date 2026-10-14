"""
Собирает данные для сайта (index.html в корне проекта) в один файл site/data.json.

Сайт построен вокруг коллажей из docs/media: каждый коллаж — отдельный раздел,
а под ним — диалоги, промпты, оценки и замечания именно тех прогонов, из которых
этот коллаж собран. Тексты вопросов и выводов — краткий пересказ docs/results.md.

Запуск из корня проекта:
    python site/build_data.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "site" / "data.json"

TEAM = ["Сухарев Александр", "Харунжин Максим", "Лемещенко Мария", "Цапко Нелли"]

ARTISTS = {
    "A": {"engine": "SDXL-Turbo", "lang": "промпт на английском", "where": "open-weight модель с HuggingFace, работает локально"},
    "B": {"engine": "Yandex ART", "lang": "промпт на русском", "where": "облако Яндекса"},
    "C": {"engine": "Yandex ART", "lang": "промпт на английском", "where": "облако Яндекса"},
}

EXPERT_NAMES = {
    "cinematic": "кинематографист", "baroque": "любитель барокко", "child": "ребёнок 6 лет",
    "minimalist": "минималист", "peredvizhnik": "передвижник",
}

SECTIONS = [
    {
        "id": "telephone",
        "kind": "telephone",
        "title": "Испорченный телефон",
        "question": "Сколько смысла переживает пересказ через картинку?",
        "how": "Художник рисует по идее. Слепой зритель смотрит на картину, придумывает ей название и описывает. "
               "Следующий художник рисует только по этому описанию: ни идеи, ни самой картины он не видит. "
               "И так шесть раз. Судья на каждом шаге оценивает, насколько картина ещё передаёт исходную идею (0–10).",
        "collages": [
            {"file": "telephone_lighthouse.jpg", "chain": "01"},
            {"file": "telephone_baba_yaga.jpg", "chain": "02"},
            {"file": "telephone_bear_skin.jpg", "chain": "03"},
        ],
        "verdict": [
            "Нейтральная идея переживает все пересказы: «Маяк в шторм» держится на 9–10 у всех трёх художников, меняются только детали и стиль.",
            "Культурный образ сохраняется только на русском промпте. У художника B «Избушка на курьих ножках» доживает до конца, "
            "у C (тот же генератор, но английский промпт) Баба-Яга превращается в «Raven Watchman», а SDXL с первого кадра рисует западного волшебника.",
            "Поговорка умирает на первом же пересказе: рисуют буквальных охотников и медведя, и зритель видит уже другую поговорку — «Медведь на ухо наступил».",
        ],
    },
    {
        "id": "culture",
        "kind": "gallery",
        "title": "Культурная асимметрия",
        "question": "Передаёт ли генератор русские культурные реалии и важен ли язык промпта?",
        "how": "Культурные идеи и нейтральная для контроля. Три художника, у каждого полный диалог с экспертом. "
               "B и C рисуют одним генератором, но на разных языках. На коллаже — лучшая картина каждого художника по каждой идее.",
        "collages": [{"file": "culture_gallery.jpg"}],
        "columns": [  # (подпись, прогон) — в том порядке, что на коллаже
            ("Масленица на деревенской площади", "culture/01/01_isolated"),
            ("Баба-Яга у избушки на курьих ножках", "culture/02/02_isolated"),
            ("Двор хрущёвки зимним вечером", "culture/03/01_isolated"),
            ("Мороз и солнце; день чудесный", "culture/04/02_isolated"),
            ("Маяк в шторм", "culture/05/01_isolated"),
        ],
        "verdict": [
            "Генератор важнее языка: на картинах SDXL зритель не узнал Масленицу ни разу, а нейтральные идеи узнавал всегда.",
            "При полном диалоге язык почти не важен: английские промпты переносят культуру транслитерацией («Baba Yaga», «Maslenitsa») и деталями («chicken legs», «pancakes»).",
            "Цитату нарисовать нельзя: «Мороз и солнце; день чудесный» не узнал никто — видна зима, но не Пушкин.",
        ],
    },
    {
        "id": "clients",
        "kind": "gallery",
        "title": "Пять заказчиков",
        "question": "Доходит ли вкус заказчика до картины через диалог?",
        "how": "Одна идея — «Маяк в шторм» — у пяти экспертов с разными вкусами. Художники A и B, два круга критики. "
               "На коллаже — лучшая картина каждого художника у каждого заказчика.",
        "collages": [{"file": "clients_gallery.jpg"}],
        "columns": [
            ("любитель барокко", "clients/baroque"),
            ("ребёнок 6 лет", "clients/child"),
            ("кинематографист", "clients/cinematic"),
            ("минималист", "clients/minimalist"),
            ("передвижник", "clients/peredvizhnik"),
        ],
        "verdict": [
            "Вкус заказчика доходит до картины полностью: барокко получил золочёную раму и аллегорию Надежды с якорем, ребёнок — радугу, кита и осьминога, "
            "минималист — плоскую графику в три цвета с одним лучом, передвижник — сепию, ржавый маяк и смотрителя с фонарём.",
            "Стиль меняется даже у слабого SDXL-Turbo: эксперт задаёт его в ответах на вопросы, художник переносит в промпт.",
            "Оценки при этом невысокие (2–7): заказчик с сильным вкусом придирчив к деталям.",
        ],
    },
    {
        "id": "shared",
        "kind": "evolution",
        "title": "Общий контекст",
        "question": "Главный вопрос работы: помогает ли художнику видеть, о чём эксперт договорился с другим художником?",
        "how": "Художники A и B рисуют «Маяк в шторм» три круга подряд. В режиме «общий контекст» каждый видит вопросы другого художника "
               "и ответы эксперта на них. На коллаже — один такой прогон: как картина меняется под критикой эксперта.",
        "collages": [{"file": "evolution_lighthouse_shared.jpg"},
                     {"file": "timelapse_lighthouse.gif", "caption": "Та же эволюция в виде анимации"}],
        "run": "experiment_lighthouse/01_shared",
        "numbers": {
            "caption": "Итог серии: 60 картин, 5 прогонов на режим, каждую оценивает панель из трёх экспертов",
            "head": ["Художник", "Раздельный контекст", "Общий контекст", "Разница"],
            "rows": [["A · SDXL-Turbo", "3.60", "2.93", "−0.67 ± 0.30"],
                     ["B · Yandex ART", "5.87", "6.07", "+0.20 ± 0.42"]],
        },
        "verdict": [
            "Общий контекст не помогает. У Yandex ART разница в пределах шума.",
            "SDXL с общим контекстом даже хуже: длинный бриф не влезает в 77 токенов его промпта, на каждую деталь остаётся меньше места.",
            "Первая серия (по 2 прогона) показывала обратное, но на 60 картинах это не повторилось. Урок: при двух повторах выводы делать нельзя.",
        ],
    },
    {
        "id": "retriever",
        "kind": "evolution",
        "title": "Восемь кругов критики",
        "question": "Становится ли картина лучше, если долго её критиковать?",
        "how": "«Золотистый ретривер»: художники A и B восемь раз подряд получают замечания эксперта и перерисовывают.",
        "collages": [{"file": "evolution_retriever.jpg"}],
        "run": "20261013_221703",
        "verdict": [
            "Нет: итерации ведут себя как пересэмплирование, а не как обучение. По всем прогонам последняя картина лучше первой в 38% случаев, хуже — в 35%.",
            "Дело не в забывчивости художника и почти не в строгости эксперта: это проверено отдельными экспериментами.",
            "SDXL плохо слушается указаний (33–62%), а Yandex ART выполняет конкретные указания почти всегда (91–100%). "
            "Его плато задают вкусовые требования эксперта и шум самой оценки: одна и та же картина у одного эксперта получила 4, 9 и 5.",
        ],
    },
    {
        "id": "moderation",
        "kind": "story",
        "title": "Модерация",
        "question": "Почему Yandex ART отказывался рисовать Масленицу?",
        "collages": [{"file": "maslenitsa_rephrased.jpg", "caption": "Масленица после переписывания промпта"}],
        "verdict": [
            "Сначала казалось, что дело в сжигании чучела. Пошаговая проверка показала другое: отказ вызывает церковь. "
            "«Церковь в дымке» и «church in the haze» отклоняются на обоих языках, а та же сцена без церкви проходит.",
            "Фильтр читает слова, а не картину: «храм вдали» проходит, «церковь» — нет. «Белая грудь» у лисы — отказ, «white chest» — нарисовано.",
            "Решение: после отказа художник сам переписывает промпт, сохраняя смысл. «Горит чучело» стало «соломенной фигурой с лентами», "
            "церковь исчезла — и Масленица узнаётся.",
        ],
    },
]

# Результаты, для которых нет коллажа, — короткими карточками в конце
EXTRA = [
    {"title": "Нейро-Крокодил",
     "text": "Шесть поговорок нарисовали без надписей, слепой зритель угадывал. Узнали только «Как слон в посудной лавке». "
             "«Рукописи не горят» зритель прочитал как «Burning the Manuscript», «Вешать лапшу на уши» — как «Spaghetti Western»."},
    {"title": "Слабые места генераторов",
     "text": "По 481 замечанию эксперта: SDXL-Turbo чаще всего теряет заказанные объекты, Yandex ART рисует всё, но промахивается с расположением и светом."},
    {"title": "Опытный художник",
     "text": "Художнику рассказали о типичных промахах его генератора. Слабому SDXL помогло: на лисе первая оценка выросла с 4.5 до 7.5."},
    {"title": "Модель верит тексту больше, чем картинке",
     "text": "С картинкой и ложным описанием («собака смотрит влево») эксперт повторил ошибку описания. Без описания — ни разу из трёх."},
]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def image_rel(raw, run_dir: Path):
    """Путь к картинке из лога → путь к JPEG на сайте относительно site/img (или site/preview)."""
    if not raw:
        return None
    name = re.split(r"[\\/]", raw)[-1]
    return str((run_dir / name).relative_to(RUNS).with_suffix(".jpg")).replace("\\", "/")


def short_error(error):
    if not error or error == "None":
        return None
    match = re.search(r"'message': '([^']+)'", str(error))
    return ("Модерация Yandex ART отказала: «" + match.group(1) + "»") if match else str(error)[:200]


def load_run(rel: str) -> dict:
    run_dir = RUNS / rel
    log = read(run_dir / "log.json")
    recovery = read(run_dir / "recovery.json") if (run_dir / "recovery.json").exists() else {}
    history = {}
    for artist, records in log["history"].items():
        items = []
        for r in records:
            review = r["review"] if isinstance(r.get("review"), dict) else r
            raw = r.get("image_path") or r.get("image")
            seen = recovery.get(re.split(r"[\\/]", raw)[-1]) if raw else None
            items.append({
                "iter": int(r["iteration"]),
                "img": image_rel(raw, run_dir),
                "score": review.get("score"),
                "prompt": r.get("prompt", ""),
                "reasoning": r.get("artist_reasoning", ""),
                "problems": review.get("problems") or [],
                "instructions": review.get("instructions") or "",
                "error": short_error(r.get("error")),
                "seen": {"title": seen["title"], "recognized": seen.get("idea_recognized"),
                         "conveyed": seen.get("details_conveyed")} if seen else None,
            })
        history[artist] = items
    return {
        "idea": log["idea"],
        "expert": EXPERT_NAMES.get(log.get("expert") or "cinematic"),
        "taste": (log.get("expert_taste") or "").strip(),
        "shared": bool(log.get("shared_questions")),
        "sees": bool(log.get("expert_sees_image")),
        "dialogue": [[d.get("artist_id") or d.get("artist") or "", d.get("question") or d.get("q") or "",
                      d.get("answer") or d.get("a") or ""] for d in log.get("dialogue", [])],
        "history": history,
    }


def best(items):
    """Лучшая картина художника — по тому же правилу, что и при сборке коллажа."""
    drawn = [i for i in items if i["img"]]
    return max(drawn, key=lambda i: i["score"]) if drawn else (items[-1] if items else None)


def load_chains(number: str) -> list:
    chains = []
    for artist in "ABC":
        run_dir = RUNS / "telephone" / number / artist
        chain = read(run_dir / "chain.json")
        chains.append({
            "artist": artist,
            "steps": [{
                "step": s["step"], "told": s["told"], "prompt": s["prompt"],
                "img": image_rel(s.get("image"), run_dir),
                "title": s["seen_title"], "desc": s["seen_description"],
                "sim": s["similarity"], "lost": s["lost"], "error": short_error(s.get("error")),
            } for s in chain["steps"]],
        })
    return chains


def build() -> dict:
    sections = []
    for spec in SECTIONS:
        section = {k: v for k, v in spec.items() if k not in ("columns", "run")}
        section["collages"] = [dict(c) for c in spec["collages"]]
        if spec["kind"] == "telephone":
            for collage in section["collages"]:
                collage["idea"] = read(RUNS / "telephone" / collage["chain"] / "A" / "chain.json")["idea"]
                collage["chains"] = load_chains(collage["chain"])
        elif spec["kind"] == "gallery":
            columns = []
            for label, rel in spec["columns"]:
                run = load_run(rel)
                run["label"] = label
                run["best"] = {a: best(items) for a, items in run["history"].items()}
                columns.append(run)
            section["columns"] = columns
        elif spec["kind"] == "evolution":
            section["run"] = load_run(spec["run"])
        sections.append(section)
    return {"team": TEAM, "artists": ARTISTS, "sections": sections, "extra": EXTRA}


def main() -> None:
    data = build()
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"site/data.json: {len(data['sections'])} разделов, {OUT.stat().st_size / 1e3:.0f} КБ")


if __name__ == "__main__":
    main()
